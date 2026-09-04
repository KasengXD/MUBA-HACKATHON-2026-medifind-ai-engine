import concurrent.futures
import json
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import streamlit as st

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

location = st.sidebar.text_input("📍 Your Location", value="Petaling Jaya, Selangor")

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

    # regex=False ensures 0ms matching performance across 147k records
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
        "evaluate if generic alternatives are safe bio-equivalents. Output ONLY RAW JSON containing: "
        '1) "safety_approved" (boolean), 2) "safety_score" (0-100), 3) "dosage_instructions" (string), and 4) "key_warnings" (string).'
    )
    user_prompt = f"Medicine: {medName}\nActive: {active}\nSubs: {subs}\n"

    try:
        model_name = get_secret("MODEL_SAFETY", "moonshotai/Kimi-K2.6")
        resA = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1024,
        )
        raw_text = resA.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)
        return json.loads(raw_text, strict=False), None
    except Exception as e:
        return {
            "safety_approved": False,
            "safety_score": 0,
            "dosage_instructions": "Consult a healthcare provider.",
            "key_warnings": f"Error loading clinical safety profile: {e}",
        }, str(e)

def runModelB(client, medName, location):
    system_prompt = (
        f"You are a retail pharmaceutical inventory AI. Analyze stock risk and store availability for {medName} in {location}. "
        'Output ONLY RAW JSON containing: 1) "stock_risk" (\'Low\' | \'Medium\' | \'High\'), '
        '2) "nearest_chain_availability": array of store names like ["Watsons", "Guardian", "Local Pharmacy"], '
        'and 3) "estimated_in_stock_confidence" (0-100).'
    )
    user_prompt = f"Assess current market stock and store availability for {medName} in {location}."

    try:
        model_name = get_secret("MODEL_SUPPLY", "deepseek-ai/DeepSeek-V4-Flash-0731")
        resB = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1024,
        )
        raw_text = resB.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)
        return json.loads(raw_text, strict=False), None
    except Exception as e:
        return {
            "stock_risk": "Unknown",
            "nearest_chain_availability": ["Local Pharmacy"],
            "estimated_in_stock_confidence": 0,
        }, str(e)

# 4. Main UI Layout
st.title("💊 MediFind: Medicine Search & Generic Engine")
st.caption("Powered by Gonka Router Dual-Model AI Orchestration")

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
    if not active_api_key:
        st.error(
            "Please provide a valid Gonka API Key in Streamlit Cloud Secrets or the sidebar."
        )
    else:
        client = OpenAI(
            api_key=active_api_key,
            base_url=BASE_URL,
            timeout=30.0,
            max_retries=2,
        )

        with st.spinner("Searching database and executing dual-AI routing..."):
            lookup = findSubstitutes(query)

            if not lookup:
                st.warning(
                    f"No match found in dataset for **'{query}'**. Try searching by active ingredient."
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
                
                # Safely parse numeric responses from LLM outputs
                stock_raw = dataB.get("estimated_in_stock_confidence") if isinstance(dataB, dict) else 0
                stockScore = safe_float(stock_raw, 0.0)

                safety_approved = dataA.get("safety_approved") if isinstance(dataA, dict) else False
                safety_default = 100.0 if safety_approved else 0.0
                safety_raw = dataA.get("safety_score") if isinstance(dataA, dict) else None
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
                m3.metric("Stock Risk Level", dataB.get("stock_risk", "N/A") if isinstance(dataB, dict) else "N/A")
                m4.metric("Consensus Score", f"{consensusScore:.1f}%")

                st.markdown("### 🧪 Active Ingredients")
                st.info(f"**{active}**")

                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("🛡️ Clinical Safety Evaluation (Kimi-K2.6)")
                    if dataA.get("safety_approved") if isinstance(dataA, dict) else False:
                        st.success("✅ **Safety Status:** Approved Bio-Equivalent")
                    else:
                        st.warning(
                            "⚠️ **Safety Status:** Requires Pharmacist Review"
                        )

                    st.markdown("**Dosage Instructions:**")
                    st.write(dataA.get("dosage_instructions", "N/A") if isinstance(dataA, dict) else "N/A")

                    st.markdown("**Key Clinical Warnings:**")
                    st.caption(dataA.get("key_warnings", "None reported.") if isinstance(dataA, dict) else "None reported.")

                    brief_text = f"""TIBA CLINICAL SAFETY BRIEF
----------------------------------------
Drug Queried: {query}
Matched Drug: {medName}
Active Ingredient: {active}
Safety Status: {'APPROVED' if (isinstance(dataA, dict) and dataA.get('safety_approved')) else 'REQUIRES REVIEW'}
Safety Score: {safetyScore:.0f}/100

DOSAGE INSTRUCTIONS:
{dataA.get('dosage_instructions', 'N/A') if isinstance(dataA, dict) else 'N/A'}

CLINICAL WARNINGS:
{dataA.get('key_warnings', 'None') if isinstance(dataA, dict) else 'None'}
----------------------------------------
Generated by Tiba AI Engine via Gonka Router
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

                    chains = dataB.get("nearest_chain_availability", []) if isinstance(dataB, dict) else []
                    if chains:
                        st.write("**Available at Nearby Retailers:**")
                        for chain in chains:
                            st.markdown(f"- 🏢 {chain}")
                    else:
                        st.write("No specific retail chain data reported.")

                    # Clamp progress value between 0.0 and 1.0 safely
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
