import concurrent.futures
import json
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(page_title = "Tiba | Generic Medecine Engine", page_icon = "💊", layout = "wide")

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GONKA_API_KEY")
BASE_URL = os.getenv("GONKA_BASE_URL", "https://api.gonkarouter.io/v1")
DATASET_PATH = os.getenv("DATASET_PATH", "cleaned_medicines_final.csv")

# 2. Data Loading with Caching
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset from {path}: {e}")
        return pd.DataFrame(columns = ["Name", "Contains"])

df = load_data(DATASET_PATH)

BRAND_ALIASES = {
    "panadol": "paracetamol",
    "tylenol": "paracetamol",
    "advil": "ibuprofen",
    "nurofen": "ibuprofen",
    "lipitor": "atorvastatin",
    "glucophage": "metformin",
}

# 3. Core Engine Functions
def findSubstitutes(searchTerm, top_n = 5):
    raw_query = str(searchTerm or "").strip()
    clean_query = raw_query.lower()

    if clean_query in BRAND_ALIASES:
        clean_query = BRAND_ALIASES[clean_query]

    match = df[df["Name"].str.contains(clean_query, case = False, na=False) | 
               df["Contains"].str.contains(clean_query, case = False, na=False)]

    if match.empty:
        return None

    match = match.copy()
    match["is_injection"] = match["Name"].str.contains("Injection|Infusion|IV", case = False, na=False)
    match["ingredient_count"] = match["Contains"].str.count(r"\+")
    sorted_matches = match.sort_values(by = ["is_injection", "ingredient_count"])

    target = sorted_matches.iloc[0]["Name"]
    active = sorted_matches.iloc[0]["Contains"]
    subs = df[(df["Contains"] == active) & (df["Name"] != target)]["Name"].unique()

    return {
        "matchedMedicine": target,
        "activeIngredient": active,
        "substitutes": list(subs[:top_n]),
        "genericMatchFound": len(subs) > 0,
    }

def runModelA(client, medName, active, subs):
    system_prompt = ("You are an expert AI clinical pharmacist. Given a queried medicine and its active ingrdients,"
                     "evaluate if generic alternatives are safe bio-equivalents. Output ONLY RAW JSON containing:"
                     '1) "safety_approved" (boolean), 2) "safety_score" (0-100), 3) "dosage_instructions" (string), and 4) "key_warnings" (string).')
    
    user_prompt = f"Medicine: {medName}\nActive: {active}\nSubs: {subs}\n"

    try:
        resA = client.chat.completions.create(model = os.getenv("MODEL_SAFETY", "moonshotai/Kimi-K2.6"), messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], response_format = {"type": "json_object"}, max_tokens = 1024,)
        

        raw_text = resA.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)
        return json.loads(raw_text, strict = False), None
    except Exception as e:
        return {
            "safety_approved": False,
            "safety_score": 0,
            "dosage_instructions": "Consult a healthcare provider.",
            "key_warnings": f"Error loading clinical safety profile: {e}",
        }, str(e)

def runModelB(client, medName, location):
    system_prompt = (f"You are a retail pharmaceutical iventory AI. Analyze stock risk and store availability for {medName} in {location}."
                     'Output ONLY RAW JSON containing: 1) "stock_risk" (\'Low\' | \'Medium\' | \'High\'),'
                     '2) "nearest_chain_availability": array of store names like ["Watsons", "Guardian", "Local Pharmacy"],'
                     'and 3) "estimated_in_stock_confidence" (0-100).')

    user_prompt = f"Assess current market stock and store availability for {medName} in {location}."

    try:

        resB = client.chat.completions.create(model = os.getenv("MODEL_SUPPLY", "deepseek-ai/DeepSeek-V4-Flash-0731"), messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], response_format = {"type": "json_object"}, max_tokens = 1024,)

        raw_text = resB.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

        if json_match:
            raw_text = json_match.group(0)
        return json.loads(raw_text, strict = False), None

    except Exception as e:
        return {
            "stock_risk": "Unknown",
            "nearest_chain_availability": ["Local Pharmacy"],
            "estimated_in_stock_confidence": 0,
        }, str(e)

# 4. Sidebar Controls
st.sidebar.title("⚙️ Engine Controls")
location = st.sidebar.text_input("📍 Your Location", value = "Petaling Jaya, Selangor")
user_api_key = st.sidebar.text_input("🔑 Gonka API Key", value = API_KEY if API_KEY else "", 
                                     type = "password", help = "Provide via .env or entered manually here.",
                                     )
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Database Size:** `{len(df):,}` records")
st.sidebar.markdown("**Safety Model:** `Kimi-K2.6`")
st.sidebar.markdown("**Supply Model:** `DeepSeek-V4-Flash`")

# 5. UI Layout
st.title("💊 Tiba: Medicine Search & Generic Engine")
st.caption("Powered by Gonka Router Dual-Model AI Orchestration")

query = st.text_input("🔍 Search Brand or Active Ingredient", placeholder = "e.g. Panadol, Augmentin, Metformin, Amoxicillin")
if query:
    if not user_api_key:
        st.error("Please provide a valid Gonka API Key in `.env` or the sidebar to run clinical AI evaluation.")
    else:
        client = OpenAI(api_key = user_api_key, base_url = BASE_URL)
        with st.spinner("Searching database and executing dual-AI routing..."):
            lookup = findSubstitutes(query)
            if not lookup:
                st.warning(f"No match found in dataset for **'{query}'**. Try searching by active ingredient.")
            else:
                medName = lookup["matchedMedicine"]
                active = lookup["activeIngredient"]
                subs = lookup["substitutes"]

                # Concurrent Model Execution
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    f_a = executor.submit(runModelA, client, medName, active, subs)
                    f_b = executor.submit(runModelB, client, medName, location)
                    dataA, errA = f_a.result()
                    dataB, errB = f_b.result()

                # Calculate Scores
                genericMatchPTS = 100 if lookup["genericMatchFound"] else 0
                stockScore = float(dataB.get("estimated_in_stock_confidence", 0))
                safetyScore = float(dataA.get("safety_score", 100 if dataA.get("safety_approved") else 0))
                consensusScore = (genericMatchPTS * 0.4) + (stockScore * 0.4) + (safetyScore * 0.2)
                st.markdown("---")

                # Header Overview Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Matched Drug", medName)
                m2.metric("Generics Found", len(subs))
                m3.metric("Stock Risk Level", dataB.get("stock_risk", "N/A"))
                m4.metric("Consensus Score", f"{consensusScore:.1f}%")

                st.markdown("###🧪 Active Ingredients")
                st.info(f"**{active}**")
                col_left, col_right = st.columns(2)

                # Clinical Safety Card (Model A)
                with col_left:
                    st.subheader("🛡️ Clinical Safety Evaluation (Kimi-K2.6)")
                    if dataA.get("safety_approved"):
                        st.success("✅ **Safety Status:** Approved Bio-Equivalent")
                    else:
                        st.warning("⚠️ **Safety Status:** Requires Pharmacist Review")

                    st.write("**Dosage Instructions:**")
                    st.write(dataA.get("dosage_instructions", "N/A"))
                    st.write("**Key Clinical Warnings:**")
                    st.caption(dataA.get("key_warnings", "None reported."))

                # Retail Supply Card (Model B)
                with col_right:
                    st.subheader("🏪 Inventory & Store Mapping (DeepSeek-V4)")
                    st.write(f"**Location Filter:** {location}")
                    chains = dataB.get("nearest_chain_availability", [])
                    if chains:
                        st.write("**Available at Nearby Retailers:**")
                        for chain in chains:
                            st.markdown(f"- 🏢 {chain}")
                    else:
                        st.write("No specific retail chain data reported.")
                    st.progress(int(stockScore) / 100, text = f"Estimated Stock Availability: {stockScore:.0f}%")

                # Generic Substitutes Table
                st.markdown("---")
                st.subheader("🔄 Verified Generic Substitutes")
                if subs:
                    sub_df = pd.DataFrame({
                        "Generic Brand Name": subs,
                        "Active Ingredient Match": [active] * len(subs),
                        "Form": ["Tablet / Capsule" if "Tablet" in s or "Capsule" in s else "Other" for s in subs],
                        "Status": ["Bio-Equivalent" for _ in subs]
                    })
                    st.dataframe(sub_df, use_container_width = True)
                else:
                    st.info("No lower-cost direct generic matches available in the local database.")



