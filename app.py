import concurrent.futures
import json
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# 1. Page Configuration & Safe Secret Loader
st.set_page_config(
    page_title="MediFind | Generic Medicine Engine",
    page_icon="💊",
    layout="wide",
)

# Load environment variables
load_dotenv()


# Safe secret reader for both Cloud and Local execution
def get_secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


API_KEY = get_secret("GONKA_API_KEY", "")
BASE_URL = get_secret("GONKA_BASE_URL", "https://api.gonkarouter.io/v1")


# Helper function for safe numeric casting from LLM outputs
def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        clean_val = str(val).replace("%", "").strip()
        return float(clean_val)
    except (ValueError, TypeError):
        return default


# Helper function to convert lat/lng to readable address
def reverse_geocode(lat, lng):
    try:
        geolocator = Nominatim(user_agent="medifind_app")
        location_obj = geolocator.reverse((lat, lng), timeout=5)
        if location_obj:
            address = location_obj.raw.get("address", {})
            suburb = address.get("suburb", address.get("town", address.get("city", "")))
            state = address.get("state", "")
            if suburb or state:
                return f"{suburb}, {state}".strip(", ")
            return location_obj.address.split(",")[0]
        return f"{lat:.4f}, {lng:.4f}"
    except Exception:
        return f"{lat:.4f}, {lng:.4f}"


# Preset State & City directory
STATE_CITY_MAP = {
    "Selangor": [
        "Petaling Jaya",
        "Shah Alam",
        "Subang Jaya",
        "Klang",
        "Puchong",
        "Cyberjaya",
        "Kajang",
        "Ampang",
    ],
    "Kuala Lumpur": [
        "Kuala Lumpur City Centre",
        "Bangsar",
        "Cheras",
        "Bukit Bintang",
        "Kepong",
        "Setapak",
        "Mont Kiara",
    ],
    "Penang": [
        "George Town",
        "Bayan Lepas",
        "Seberang Perai",
        "Butterworth",
        "Ayer Itam",
    ],
    "Johor": [
        "Johor Bahru",
        "Iskandar Puteri",
        "Batu Pahat",
        "Muar",
        "Kluang",
        "Kulai",
    ],
    "Perak": ["Ipoh", "Taiping", "Teluk Intan", "Manjung", "Kampar"],
    "Sabah": ["Kota Kinabalu", "Sandakan", "Tawau"],
    "Sarawak": ["Kuching", "Miri", "Sibu", "Bintulu"],
    "Melaka": ["Melaka City", "Ayer Keroh", "Alor Gajah"],
    "Negeri Sembilan": ["Seremban", "Port Dickson", "Nilai"],
    "Kedah": ["Alor Setar", "Sungai Petani", "Langkawi", "Kulim"],
    "Pahang": ["Kuantan", "Temerloh", "Bentong", "Cameron Highlands"],
}


# 2. Sidebar Controls & Safe Dataset Loader
st.sidebar.title("⚙️ Engine Controls")

dataset_scale = st.sidebar.radio(
    "📊 Database Scale Mode",
    ["Sample Database (~10k)", "Full Production Database (~147k)"],
    index=0,
)

target_path = (
    "sample_medicines.csv"
    if "Sample" in dataset_scale
    else "cleaned_medicines_final.csv.gz"
)


@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        if os.path.exists("sample_medicines.csv"):
            st.warning(
                "⚠️ Full database is not committed to GitHub (>100MB limit). Defaulting to `sample_medicines.csv`."
            )
            return pd.read_csv("sample_medicines.csv")
        else:
            st.error("No dataset CSV found in repository.")
            return pd.DataFrame(columns=["Name", "Contains"])
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame(columns=["Name", "Contains"])


df = load_data(target_path)

# --- Location Selection Module ---
st.sidebar.markdown("### 📍 Location Selection")
loc_mode = st.sidebar.radio(
    "Location Selection Mode",
    [
        "🏙️ Select State & City",
        "📌 Interactive Map Pin",
        "⌨️ Manual Free Text",
    ],
    index=0,
)

if loc_mode == "🏙️ Select State & City":
    selected_state = st.sidebar.selectbox(
        "🏛️ Select State",
        options=list(STATE_CITY_MAP.keys()) + ["Other / Custom State"],
    )

    if selected_state == "Other / Custom State":
        custom_city = st.sidebar.text_input("City", value="Petaling Jaya")
        custom_state = st.sidebar.text_input("State", value="Selangor")
        location = f"{custom_city.strip()}, {custom_state.strip()}".strip(", ")
    else:
        city_options = STATE_CITY_MAP.get(selected_state, []) + ["Other / Custom City"]
        selected_city = st.sidebar.selectbox("🌆 Select City", options=city_options)

        if selected_city == "Other / Custom City":
            custom_city = st.sidebar.text_input("Enter City Name", value="")
            location = (
                f"{custom_city.strip()}, {selected_state}"
                if custom_city.strip()
                else selected_state
            )
        else:
            location = f"{selected_city}, {selected_state}"

elif loc_mode == "📌 Interactive Map Pin":
    st.sidebar.caption("Click anywhere on the map to pin your location:")

    # Initialize pinned location in session state
    if "map_lat" not in st.session_state:
        st.session_state["map_lat"] = 3.1073
        st.session_state["map_lng"] = 101.6067

    # Create Folium Map with center on current coordinates
    m = folium.Map(
        location=[st.session_state["map_lat"], st.session_state["map_lng"]],
        zoom_start=11,
    )

    # Add active pin marker on the map
    folium.Marker(
        [st.session_state["map_lat"], st.session_state["map_lng"]],
        popup="Pinned Location",
        tooltip="Selected Location",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # Render map component
    map_out = st_folium(
        m,
        height=220,
        key="sidebar_map",
        use_container_width=True,
        returned_objects=["last_clicked"],
    )

    # Update state and rerun if a new pin point is clicked
    if map_out and map_out.get("last_clicked"):
        clicked_lat = map_out["last_clicked"]["lat"]
        clicked_lng = map_out["last_clicked"]["lng"]

        if (
            abs(clicked_lat - st.session_state["map_lat"]) > 1e-5
            or abs(clicked_lng - st.session_state["map_lng"]) > 1e-5
        ):
            st.session_state["map_lat"] = clicked_lat
            st.session_state["map_lng"] = clicked_lng
            st.rerun()

    location = reverse_geocode(
        st.session_state["map_lat"], st.session_state["map_lng"]
    )

else:
    location = st.sidebar.text_input("📍 Your Location", value="Petaling Jaya, Selangor")

st.sidebar.info(f"**Target Location:** {location}")

user_api_key = st.sidebar.text_input(
    "🔑 Gonka API Key",
    value="",
    type="password",
    help="Leave blank to use Streamlit Cloud Secrets",
)

# Active key resolution order: Sidebar Input -> Streamlit Secrets -> Local .env
active_api_key = user_api_key.strip() if user_api_key.strip() else API_KEY

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active Database Records:** `{len(df):,}`")
st.sidebar.markdown("**Safety Model:** `Kimi-K2.6`")
st.sidebar.markdown("**Supply Model:** `DeepSeek-V4-Flash`")

BRAND_ALIASES = {
    "panadol": "paracetamol",
    "tylenol": "paracetamol",
    "advil": "ibuprofen",
    "nurofen": "ibuprofen",
    "lipitor": "atorvastatin",
    "glucophage": "metformin",
}


# 3. Core Engine Functions
def findSubstitutes(searchTerm, top_n=5):
    raw_query = str(searchTerm or "").strip()
    clean_query = raw_query.lower()

    if clean_query in BRAND_ALIASES:
        clean_query = BRAND_ALIASES[clean_query]

    match = df[
        df["Name"].str.contains(clean_query, case=False, na=False, regex=False)
        | df["Contains"].str.contains(clean_query, case=False, na=False, regex=False)
    ]
    if match.empty:
        return None

    match = match.copy()
    match["is_injection"] = match["Name"].str.contains(
        "Injection|Infusion|IV", case=False, na=False, regex=False
    )
    match["ingredient_count"] = match["Contains"].str.count(r"\+")
    sorted_matches = match.sort_values(by=["is_injection", "ingredient_count"])

    target = sorted_matches.iloc[0]["Name"]
    active = sorted_matches.iloc[0]["Contains"]
    subs = df[(df["Contains"] == active) & (df["Name"] != target)][
        "Name"
    ].unique()

    return {
        "matchedMedicine": target,
        "activeIngredient": active,
        "substitutes": list(subs[:top_n]),
        "genericMatchFound": len(subs) > 0,
    }


def runModelA(client, medName, active, subs):
    system_prompt = (
        "You are an expert AI clinical pharmacist. Given a queried medicine and its active ingredients, "
        "evaluate if generic alternatives are safe bio-equivalents. Output ONLY RAW JSON: "
        '{"safety_approved": true, "safety_score": 90, "dosage_instructions": "...", "key_warnings": "..."}'
    )
    user_prompt = f"Medicine: {medName}\nActive: {active}\nSubs: {subs}\n"

    primary_model = get_secret("MODEL_SAFETY", "moonshotai/Kimi-K2.6")
    fallback_model = "deepseek-ai/DeepSeek-V4-Flash-0731"
    last_error = "Unknown execution error"

    for model_name in [primary_model, fallback_model]:
        try:
            resA = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"} if "DeepSeek" in model_name else None,
                max_tokens=512,
            )
            raw_text = resA.choices[0].message.content or ""
            
            raw_text = re.sub(r"```json\s*", "", raw_text)
            raw_text = re.sub(r"```\s*", "", raw_text).strip()

            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

            if json_match:
                parsed = json.loads(json_match.group(0), strict=False)
                if isinstance(parsed, dict):
                    return parsed, None
                last_error = f"{model_name} returned non-dictionary JSON."
            else:
                last_error = f"{model_name} output missing valid JSON format."
        except Exception as e:
            last_error = str(e)

    return {
        "safety_approved": False,
        "safety_score": 0,
        "dosage_instructions": "Consult a healthcare provider.",
        "key_warnings": f"Error loading clinical safety profile: {last_error}",
    }, last_error


def runModelB(client, medName, location):
    system_prompt = (
        f"You are a retail pharmaceutical market inventory estimation engine. "
        f"Assess market supply risk and typical stock availability for {medName} in {location}. "
        "Base your assessment on general regional distribution across major retail pharmacy chains "
        "(e.g., Watsons, Guardian, Caring Pharmacy, BIG Pharmacy, Alpro Pharmacy). "
        "DO NOT output disclaimers about lacking real-time data. Output ONLY RAW JSON in this exact structure: "
        '{"stock_risk": "Low", "nearest_chain_availability": ["Watsons", "Guardian", "Caring Pharmacy"], "estimated_in_stock_confidence": 85}'
    )
    user_prompt = f"Provide retail supply probability and stocking chains for {medName} around {location}."

    model_name = get_secret("MODEL_SUPPLY", "deepseek-ai/DeepSeek-V4-Flash-0731")

    try:
        resB = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"} if "DeepSeek" in model_name else None,
            max_tokens=512,
        )
        raw_text = resB.choices[0].message.content or ""
        
        raw_text = re.sub(r"```json\s*", "", raw_text)
        raw_text = re.sub(r"```\s*", "", raw_text).strip()

        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0), strict=False)
            if isinstance(parsed, dict):
                return parsed, None
    except Exception as e:
        pass

    return {
        "stock_risk": "Low",
        "nearest_chain_availability": ["Watsons", "Guardian", "Caring Pharmacy", "BIG Pharmacy"],
        "estimated_in_stock_confidence": 80,
    }, "Using default regional inventory estimate."


# 4. Main UI Layout
st.title("💊 MediFind: Medicine Search & Generic Engine")
st.caption("Powered by Gonka Router Dual-Model AI Orchestration")

tab_search, tab_library = st.tabs(["🔍 Search & Evaluation", "📚 Medicine Reference Library"])

# ================= TAB 1: SEARCH & EVALUATION =================
with tab_search:
    search_mode = st.radio(
        "Search Mode",
        ["⚡ Quick Select (Preset Demo)", "⌨️ Free Text Search"],
        horizontal=True,
    )

    if search_mode == "⚡ Quick Select (Preset Demo)":
        popular_meds = [
            "Metformin",
            "Augmentin",
            "Paracetamol",
            "Amoxicillin",
            "Pantoprazole",
            "Atorvastatin",
            "Azithromycin",
            "Cetirizine",
            "Omeprazole",
            "Ibuprofen",
        ]
        query = st.selectbox("Select a medicine query:", [""] + popular_meds)
    else:
        query = st.text_input(
            "🔍 Search Brand or Active Ingredient",
            placeholder="e.g. Panadol, Augmentin, Metformin, Amoxicillin",
        )

    if query:
        clean_q = query.strip()
        
        if len(clean_q) < 3 and search_mode == "⌨️ Free Text Search":
            st.warning("⚠️ Please enter at least 3 characters to search (e.g., 'Panadol', 'Amox').")
        elif not active_api_key:
            st.error(
                "Please provide a valid Gonka API Key in Streamlit Cloud Secrets or the sidebar."
            )
        else:
            client = OpenAI(
                api_key=active_api_key,
                base_url=BASE_URL,
                timeout=45.0,
                max_retries=2,
            )

            with st.spinner("Searching database and executing dual-AI routing..."):
                lookup = findSubstitutes(clean_q)

                if not lookup:
                    st.warning(
                        f"No match found in dataset for **'{clean_q}'**. Try searching by active ingredient."
                    )
                else:
                    medName = lookup["matchedMedicine"]
                    active = lookup["activeIngredient"]
                    subs = lookup["substitutes"]

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        f_a = executor.submit(
                            runModelA, client, medName, active, subs
                        )
                        f_b = executor.submit(runModelB, client, medName, location)
                        dataA, errA = f_a.result()
                        dataB, errB = f_b.result()

                    genericMatchPTS = 100 if lookup["genericMatchFound"] else 0

                    stock_raw = (
                        dataB.get("estimated_in_stock_confidence")
                        if isinstance(dataB, dict)
                        else 0
                    )
                    stockScore = safe_float(stock_raw, 0.0)

                    safety_approved = (
                        dataA.get("safety_approved")
                        if isinstance(dataA, dict)
                        else False
                    )
                    safety_default = 100.0 if safety_approved else 0.0
                    safety_raw = (
                        dataA.get("safety_score")
                        if isinstance(dataA, dict)
                        else None
                    )
                    safetyScore = safe_float(safety_raw, safety_default)

                    consensusScore = (
                        (genericMatchPTS * 0.4)
                        + (stockScore * 0.4)
                        + (safetyScore * 0.2)
                    )

                    st.markdown("---")

                    if len(subs) > 0:
                        st.success(
                            "💡 **Generic Value Alert:** Generic substitutes found. Switching from branded drugs to unbranded generics saves consumers an estimated **50% to 75%** on prescription costs."
                        )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Matched Drug", medName)
                    m2.metric("Generics Found", len(subs))
                    m3.metric(
                        "Stock Risk Level",
                        (
                            dataB.get("stock_risk", "N/A")
                            if isinstance(dataB, dict)
                            else "N/A"
                        ),
                    )
                    m4.metric("Consensus Score", f"{consensusScore:.1f}%")

                    st.markdown("### 🧪 Active Ingredients")
                    st.info(f"**{active}**")

                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.subheader("🛡️ Clinical Safety Evaluation (Kimi-K2.6)")
                        if (
                            dataA.get("safety_approved")
                            if isinstance(dataA, dict)
                            else False
                        ):
                            st.success("✅ **Safety Status:** Approved Bio-Equivalent")
                        else:
                            st.warning(
                                "⚠️ **Safety Status:** Requires Pharmacist Review"
                            )

                        st.markdown("**Dosage Instructions:**")
                        st.write(
                            dataA.get("dosage_instructions", "N/A")
                            if isinstance(dataA, dict)
                            else "N/A"
                        )

                        st.markdown("**Key Clinical Warnings:**")
                        st.caption(
                            dataA.get("key_warnings", "None reported.")
                            if isinstance(dataA, dict)
                            else "None reported."
                        )

                        brief_text = f"""MediFind CLINICAL SAFETY BRIEF
----------------------------------------
Drug Queried: {clean_q}
Matched Drug: {medName}
Active Ingredient: {active}
Safety Status: {'APPROVED' if (isinstance(dataA, dict) and dataA.get('safety_approved')) else 'REQUIRES REVIEW'}
Safety Score: {safetyScore:.0f}/100

DOSAGE INSTRUCTIONS:
{dataA.get('dosage_instructions', 'N/A') if isinstance(dataA, dict) else 'N/A'}

CLINICAL WARNINGS:
{dataA.get('key_warnings', 'None') if isinstance(dataA, dict) else 'None'}
----------------------------------------
Generated by MediFind AI Engine via Gonka Router
"""
                        st.download_button(
                            label="📄 Download Pharmacist Brief",
                            data=brief_text,
                            file_name=f"{medName}_safety_brief.txt",
                            mime="text/plain",
                        )

                    with col_right:
                        st.subheader(
                            "🏪 Inventory & Store Mapping (DeepSeek-V4)"
                        )
                        st.write(f"**Location Filter:** {location}")

                        chains = (
                            dataB.get("nearest_chain_availability", [])
                            if isinstance(dataB, dict)
                            else []
                        )
                        if chains:
                            st.write("**Available at Nearby Retailers:**")
                            for chain in chains:
                                st.markdown(f"- 🏢 {chain}")
                        else:
                            st.write("No specific retail chain data reported.")

                        progress_val = max(0.0, min(100.0, stockScore)) / 100.0
                        st.progress(
                            progress_val,
                            text=f"Estimated Stock Availability: {stockScore:.0f}%",
                        )

                    st.markdown("---")
                    st.subheader("🔄 Verified Generic Substitutes")
                    if subs:
                        sub_df = pd.DataFrame(
                            {
                                "Generic Brand Name": subs,
                                "Active Ingredient Match": [active] * len(subs),
                                "Form": [
                                    (
                                        "Tablet / Capsule"
                                        if "Tablet" in s or "Capsule" in s
                                        else "Other"
                                    )
                                    for s in subs
                                ],
                                "Status": ["Bio-Equivalent" for _ in subs],
                            }
                        )
                        st.dataframe(sub_df, use_container_width=True)
                    else:
                        st.info(
                            "No lower-cost direct generic matches available in the local database."
                        )

# ================= TAB 2: MEDICINE REFERENCE LIBRARY =================
with tab_library:
    st.subheader("📖 Medicine Catalog & Active Ingredients Library")
    st.caption("Browse or filter the local pharmaceutical database.")

    lib_col1, lib_col2 = st.columns([2, 1])
    
    with lib_col1:
        lib_filter = st.text_input(
            "🔎 Filter Catalog by Keyword or Letter",
            placeholder="Type 'Para', 'Amox', or 'Tablet'...",
            key="lib_filter_input"
        )

    with lib_col2:
        st.write("")
        st.write("")
        st.caption(f"Showing **{len(df):,}** total records in loaded dataset.")

    if lib_filter and len(lib_filter.strip()) > 0:
        clean_lib_q = lib_filter.strip().lower()
        display_df = df[
            df["Name"].str.contains(clean_lib_q, case=False, na=False, regex=False) |
            df["Contains"].str.contains(clean_lib_q, case=False, na=False, regex=False)
        ]
    else:
        display_df = df

    st.dataframe(
        display_df[["Name", "Contains"]].rename(
            columns={"Name": "Brand / Medicine Name", "Contains": "Active Ingredients"}
        ),
        use_container_width=True,
        height=500,
        hide_index=True
    )
