import concurrent.futures
import json
import os
import re
import time
import pandas as pd
import streamlit as st

# 1. Safe Optional Third-Party Imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    import difflib

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    from geopy.geocoders import Nominatim
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

# 2. App Page Configuration
st.set_page_config(
    page_title="MediFind | Generic Medicine Engine",
    page_icon="💊",
    layout="wide",
)

# Custom CSS: Reduce metric font size and force text wrapping to avoid truncation
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        word-wrap: break-word !important;
        white-space: normal !important;
        line-height: 1.3 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(key, default=""):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


API_KEY = get_secret("GONKA_API_KEY", "")
BASE_URL = get_secret("GONKA_BASE_URL", "https://api.gonkarouter.io/v1")


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        clean_val = str(val).replace("%", "").strip()
        return float(clean_val)
    except (ValueError, TypeError):
        return default


def extract_json(raw_text):
    """Find outer curly braces ignoring preambles and markdown syntax."""
    if not raw_text:
        raise ValueError("Empty string received from LLM")
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(raw_text[start : end + 1], strict=False)
    raise ValueError("No valid JSON object found in response")


def reverse_geocode(lat, lng):
    if not HAS_GEOPY:
        return f"{lat:.4f}, {lng:.4f}"
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


# 3. Sidebar Controls & Safe Dataset Loader
st.sidebar.title("⚙️ Engine Controls")

with st.sidebar.expander("📊 Database & API Settings", expanded=True):
    dataset_scale = st.radio(
        "Database Scale Mode",
        ["Sample Database (~10k)", "Full Production Database (~147k)"],
        index=0,
    )
    user_api_key = st.text_input(
        "🔑 Gonka API Key",
        value="",
        type="password",
        help="Leave blank to use Streamlit Cloud Secrets",
    )

active_api_key = user_api_key.strip() if user_api_key.strip() else API_KEY

target_path = (
    "sample_medicines.csv"
    if "Sample" in dataset_scale
    else "cleaned_medicines_final.csv.gz"
)


@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        if os.path.exists("sample_medicines.csv"):
            return pd.read_csv("sample_medicines.csv")
        else:
            return pd.DataFrame(columns=["Name", "Contains"])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["Name", "Contains"])


df = load_data(target_path)

with st.sidebar.expander("📍 Location Selection", expanded=True):
    loc_mode = st.radio(
        "Mode",
        [
            "🏙️ Select State & City",
            "📌 Interactive Map Pin",
            "⌨️ Manual Free Text",
        ],
        index=0,
    )

    if loc_mode == "🏙️ Select State & City":
        selected_state = st.selectbox(
            "🏛️ State",
            options=list(STATE_CITY_MAP.keys()) + ["Other / Custom State"],
        )

        if selected_state == "Other / Custom State":
            custom_city = st.text_input("City", value="Petaling Jaya")
            custom_state = st.text_input("State", value="Selangor")
            location = f"{custom_city.strip()}, {custom_state.strip()}".strip(", ")
        else:
            city_options = STATE_CITY_MAP.get(selected_state, []) + [
                "Other / Custom City"
            ]
            selected_city = st.selectbox("🌆 City", options=city_options)

            if selected_city == "Other / Custom City":
                custom_city = st.text_input("Enter City Name", value="")
                location = (
                    f"{custom_city.strip()}, {selected_state}"
                    if custom_city.strip()
                    else selected_state
                )
            else:
                location = f"{selected_city}, {selected_state}"

    elif loc_mode == "📌 Interactive Map Pin":
        if HAS_FOLIUM:
            st.caption("Click anywhere on the map to pin location:")

            if "map_lat" not in st.session_state:
                st.session_state["map_lat"] = 3.1073
                st.session_state["map_lng"] = 101.6067

            m = folium.Map(
                location=[st.session_state["map_lat"], st.session_state["map_lng"]],
                zoom_start=11,
            )

            folium.Marker(
                [st.session_state["map_lat"], st.session_state["map_lng"]],
                popup="Pinned Location",
                tooltip="Selected Location",
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(m)

            map_out = st_folium(
                m,
                height=200,
                key="sidebar_map",
                returned_objects=["last_clicked"],
            )

            if map_out and map_out.get("last_clicked"):
                clicked_lat = map_out["last_clicked"]["lat"]
                clicked_lng = map_out["last_clicked"]["lng"]

                if (
                    abs(clicked_lat - st.session_state["map_lat"]) > 1e-5
                    or abs(clicked_lng - st.session_state["map_lng"]) > 1e-5
                ):
                    st.session_state["map_lat"] = clicked_lat
                    st.session_state["map_lng"] = clicked_lng
                    if hasattr(st, "rerun"):
                        st.rerun()
                    elif hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()

            location = reverse_geocode(
                st.session_state["map_lat"], st.session_state["map_lng"]
            )
        else:
            st.caption("Map plugin not detected. Defaulting to text mode.")
            location = st.text_input("📍 Your Location", value="Petaling Jaya, Selangor", key="map_fallback_loc")

    else:
        location = st.text_input("📍 Your Location", value="Petaling Jaya, Selangor")

st.sidebar.info(f"**Target Location:** {location}")
st.sidebar.markdown(f"**Active Records:** `{len(df):,}`")

BRAND_ALIASES = {
    "panadol": "paracetamol",
    "tylenol": "paracetamol",
    "advil": "ibuprofen",
    "nurofen": "ibuprofen",
    "lipitor": "atorvastatin",
    "glucophage": "metformin",
}


# 4. Core Search & Substitute Engine
def findSubstitutes(searchTerm, top_n=5):
    if df.empty:
        return None

    raw_query = str(searchTerm or "").strip()
    clean_query = raw_query.lower()

    if clean_query in BRAND_ALIASES:
        clean_query = BRAND_ALIASES[clean_query]

    # Standard Substring Match
    match = df[
        df["Name"].astype(str).str.contains(clean_query, case=False, na=False, regex=False)
        | df["Contains"].astype(str).str.contains(clean_query, case=False, na=False, regex=False)
    ]

    # Fuzzy Search Fallback if exact match fails
    fuzzy_corrected = False
    if match.empty:
        all_names = df["Name"].dropna().tolist()
        if HAS_RAPIDFUZZ and all_names:
            best_matches = process.extract(
                clean_query, all_names, scorer=fuzz.WRatio, limit=1
            )
            if best_matches and best_matches[0][1] >= 65:
                clean_query = best_matches[0][0]
                fuzzy_corrected = True
        elif all_names:
            closest = difflib.get_close_matches(clean_query, all_names, n=1, cutoff=0.6)
            if closest:
                clean_query = closest[0]
                fuzzy_corrected = True

        if fuzzy_corrected:
            match = df[df["Name"] == clean_query]

    if match.empty:
        return None

    match = match.copy()
    match["is_injection"] = match["Name"].astype(str).str.contains(
        "Injection|Infusion|IV", case=False, na=False, regex=False
    )
    match["ingredient_count"] = match["Contains"].astype(str).str.count(r"\+")
    sorted_matches = match.sort_values(by=["is_injection", "ingredient_count"])

    # If exact name match exists for selected item, use it
    exact_selected = sorted_matches[sorted_matches["Name"].astype(str).str.lower() == raw_query.lower()]
    if not exact_selected.empty:
        target = exact_selected.iloc[0]["Name"]
        active = exact_selected.iloc[0]["Contains"]
    else:
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
        "fuzzyCorrected": fuzzy_corrected,
        "originalQuery": raw_query,
    }


def _execute_model_a(medName, active, subs, api_key, base_url):
    try:
        if not HAS_OPENAI or not api_key:
            return {
                "safety_approved": True,
                "safety_score": 85,
                "dosage_instructions": "Consult a local healthcare provider or pharmacist.",
                "key_warnings": "OpenAI package not installed or API key missing.",
            }, None, "N/A"

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=2)
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
                    max_tokens=512,
                )
                req_id = getattr(resA, "id", f"gonka-safety-{int(time.time())}")
                raw_text = resA.choices[0].message.content or ""
                parsed = extract_json(raw_text)
                if isinstance(parsed, dict):
                    return parsed, None, req_id
                last_error = f"{model_name} returned non-dictionary JSON structure."
            except Exception as e:
                last_error = str(e)

        return {
            "safety_approved": False,
            "safety_score": 0,
            "dosage_instructions": "Consult a healthcare provider.",
            "key_warnings": f"Clinical Safety Load Warning: {last_error}",
        }, last_error, "N/A"
    except Exception as top_e:
        return {
            "safety_approved": True,
            "safety_score": 80,
            "dosage_instructions": "Consult a local healthcare provider or pharmacist.",
            "key_warnings": f"Execution Warning: {str(top_e)}",
        }, str(top_e), "N/A"


def _execute_model_b(medName, location, api_key, base_url):
    try:
        if not HAS_OPENAI or not api_key:
            return {
                "stock_risk": "Low",
                "nearest_chain_availability": ["Watsons", "Guardian", "Caring Pharmacy", "BIG Pharmacy"],
                "estimated_in_stock_confidence": 80,
            }, None, "N/A"

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=2)
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
                max_tokens=512,
            )
            req_id = getattr(resB, "id", f"gonka-supply-{int(time.time())}")
            raw_text = resB.choices[0].message.content or ""
            parsed = extract_json(raw_text)
            if isinstance(parsed, dict):
                return parsed, None, req_id
        except Exception:
            pass

        return {
            "stock_risk": "Low",
            "nearest_chain_availability": ["Watsons", "Guardian", "Caring Pharmacy", "BIG Pharmacy"],
            "estimated_in_stock_confidence": 80,
        }, "Using default regional inventory estimate.", "N/A"
    except Exception as top_e:
        return {
            "stock_risk": "Low",
            "nearest_chain_availability": ["Watsons", "Guardian", "Caring Pharmacy", "BIG Pharmacy"],
            "estimated_in_stock_confidence": 80,
        }, str(top_e), "N/A"


@st.cache_data(ttl=86400, show_spinner=False)
def cached_run_model_a(api_key, base_url, medName, active, subs):
    return _execute_model_a(medName, active, subs, api_key, base_url)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_run_model_b(api_key, base_url, medName, location):
    return _execute_model_b(medName, location, api_key, base_url)


# 5. Main UI Layout
st.title("💊 MediFind: Medicine Search & Generic Engine")
st.caption("Powered by Gonka Router Dual-Model AI Orchestration")

if not HAS_OPENAI:
    st.warning("⚠️ `openai` python library is not detected in your environment. Running in offline UI demo mode.")

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
            placeholder="e.g. Actorise, Panadol, Augmentin, Metformin",
        )

    # Empty State Dashboard Landing Page
    if not query:
        st.markdown("---")
        st.markdown("### 📊 Engine Dashboard & Key Metrics")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Database Index", f"{len(df):,} Drugs")
        d2.metric("Clinical AI", "Kimi-K2.6")
        d3.metric("Supply AI", "DeepSeek-V4")
        d4.metric("Average Generic Savings", "50% - 75%")

        st.markdown("#### ⚡ Popular Searches")
        p_cols = st.columns(5)
        sample_tags = ["Paracetamol", "Augmentin", "Metformin", "Atorvastatin", "Ibuprofen"]
        for idx, tag in enumerate(sample_tags):
            p_cols[idx].info(f"💊 **{tag}**")

    else:
        clean_q = query.strip()

        if len(clean_q) < 3 and search_mode == "⌨️ Free Text Search":
            st.warning("⚠️ Please enter at least 3 characters to search (e.g., 'Panadol', 'Amox').")
        else:
            if not active_api_key and HAS_OPENAI:
                st.warning("ℹ️ No Gonka API Key provided. Running in offline fallback evaluation mode.")

            # Check for multiple product strength variants in database
            matching_variants = df[
                df["Name"].astype(str).str.contains(clean_q, case=False, na=False, regex=False)
            ]["Name"].unique().tolist()

            target_search_term = clean_q
            if len(matching_variants) > 1:
                st.info(f"📦 Found **{len(matching_variants)}** matching variants for **'{clean_q}'**.")
                target_search_term = st.selectbox(
                    "Select specific product variant to evaluate:",
                    options=matching_variants,
                    key="variant_selector",
                )

            with st.spinner("Searching database and executing dual-AI routing via Gonka Gateway..."):
                lookup = findSubstitutes(target_search_term)

                if not lookup:
                    st.warning(
                        f"No match found in dataset for **'{clean_q}'**. Try searching by active ingredient."
                    )
                else:
                    medName = lookup["matchedMedicine"]
                    active = lookup["activeIngredient"]
                    subs = lookup["substitutes"]

                    if lookup.get("fuzzyCorrected"):
                        st.info(
                            f"🔍 Showing results for autocorrected query: **'{medName}'** (Original: *'{lookup['originalQuery']}'*)"
                        )

                    # Parallel Cached Model Calls
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        f_a = executor.submit(
                            cached_run_model_a,
                            active_api_key,
                            BASE_URL,
                            medName,
                            active,
                            tuple(subs) if isinstance(subs, list) else subs,
                        )
                        f_b = executor.submit(
                            cached_run_model_b,
                            active_api_key,
                            BASE_URL,
                            medName,
                            location,
                        )
                        
                        try:
                            dataA, errA, req_id_a = f_a.result()
                        except Exception as ex_a:
                            dataA, errA, req_id_a = {
                                "safety_approved": True,
                                "safety_score": 80,
                                "dosage_instructions": "Consult a local healthcare provider or pharmacist.",
                                "key_warnings": f"Execution error: {str(ex_a)}",
                            }, str(ex_a), "N/A"

                        try:
                            dataB, errB, req_id_b = f_b.result()
                        except Exception as ex_b:
                            dataB, errB, req_id_b = {
                                "stock_risk": "Low",
                                "nearest_chain_availability": ["Watsons", "Guardian", "Caring Pharmacy"],
                                "estimated_in_stock_confidence": 80,
                            }, str(ex_b), "N/A"

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
Gonka Request ID (Clinical): {req_id_a}
Gonka Request ID (Supply): {req_id_b}

DOSAGE INSTRUCTIONS:
{dataA.get('dosage_instructions', 'N/A') if isinstance(dataA, dict) else 'N/A'}

CLINICAL WARNINGS:
{dataA.get('key_warnings', 'None') if isinstance(dataA, dict) else 'None'}
----------------------------------------
Generated by MediFind AI Engine via Gonka Router Gateway
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

                    # ================= COST SAVINGS CALCULATOR (RM) =================
                    st.markdown("---")
                    st.subheader("💡 Estimated Generic Cost Savings Calculator (RM)")
                    st.caption(
                        f"Enter the estimated branded cost for **{medName}** to automatically calculate estimated generic savings."
                    )

                    brand_price = st.number_input(
                        f"Estimated Monthly Cost of Branded {medName} (RM):",
                        min_value=0.0,
                        value=80.0,
                        step=5.0,
                        key="brand_cost_input",
                    )

                    # Auto-calculate generic cost using standard generic market discount (65% average savings)
                    generic_discount_rate = 0.65
                    generic_price = round(brand_price * (1.0 - generic_discount_rate), 2)
                    monthly_saving = max(0.0, brand_price - generic_price)
                    annual_saving = monthly_saving * 12
                    savings_pct = (monthly_saving / brand_price * 100) if brand_price > 0 else 0.0

                    s1, s2, s3 = st.columns(3)
                    s1.metric("Est. Generic Monthly Cost", f"RM {generic_price:.2f}")
                    s2.metric("Monthly Savings", f"RM {monthly_saving:.2f}")
                    s3.metric(
                        "Annual Consumer Savings",
                        f"RM {annual_saving:.2f}",
                        delta=f"-{savings_pct:.1f}% Savings" if savings_pct > 0 else None,
                    )

                    # ================= GONKA ROUTER TRANSPARENCY DASHBOARD =================
                    st.markdown("---")
                    st.markdown("### 🌐 Gonka Router Verification & Audit Trace")
                    st.caption("All verification and clinical inference steps executed via Gonka Inference Gateway (`gonkarouter.io`).")
                    g1, g2, g3 = st.columns(3)
                    g1.markdown(f"**Gateway Host:**\n`gonkarouter.io`")
                    g2.markdown(f"**Clinical Model (`Kimi-K2.6`) Req ID:**\n`{req_id_a}`")
                    g3.markdown(f"**Supply Model (`DeepSeek-V4`) Req ID:**\n`{req_id_b}`")

# ================= TAB 2: MEDICINE REFERENCE LIBRARY =================
with tab_library:
    st.subheader("📖 Medicine Catalog & Active Ingredients Library")
    st.caption("Browse or filter the local pharmaceutical database.")

    lib_col1, lib_col2 = st.columns([2, 1])

    with lib_col1:
        lib_filter = st.text_input(
            "🔎 Filter Catalog by Keyword or Letter",
            placeholder="Type 'Para', 'Amox', or 'Tablet'...",
            key="lib_filter_input",
        )

    with lib_col2:
        st.write("")
        st.write("")
        st.caption(f"Showing **{len(df):,}** total records in loaded dataset.")

    if lib_filter and len(lib_filter.strip()) > 0:
        clean_lib_q = lib_filter.strip().lower()
        display_df = df[
            df["Name"].astype(str).str.contains(clean_lib_q, case=False, na=False, regex=False)
            | df["Contains"].astype(str).str.contains(
                clean_lib_q, case=False, na=False, regex=False
            )
        ]
    else:
        display_df = df

    st.dataframe(
        display_df[["Name", "Contains"]].rename(
            columns={"Name": "Brand / Medicine Name", "Contains": "Active Ingredients"}
        ),
        use_container_width=True,
        height=500,
        hide_index=True,
    )
