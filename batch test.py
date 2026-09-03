
import concurrent.futures
import json
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("GONKA_API_KEY")


if not API_KEY or not API_KEY.startswith("sk-"):
    raise ValueError(
        "Invalid or missing API key! Please set GONKA_API_KEY in your .env file."
    )

client = OpenAI(
    api_key=API_KEY, base_url=os.getenv("GONKA_BASE_URL", "https://api.gonkarouter.io/v1")
)

print("Loading dataset...")

try:
    df = pd.read_csv(os.getenv("DATASET_PATH", "cleaned_medicines_final.csv"))
    print(f"Loaded dataset: {len(df)} records.\n")
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.DataFrame(columns=["Name", "Contains"])

# Place BRAND_ALIASES map right above findSubstitutes
BRAND_ALIASES = {
    "panadol": "paracetamol",
    "tylenol": "paracetamol",
    "advil": "ibuprofen",
    "nurofen": "ibuprofen",
    "lipitor": "atorvastatin",
    "glucophage": "metformin",
}


def findSubstitutes(searchTerm, top_n=5):
    raw_query = str(searchTerm or "").strip()
    clean_query = raw_query.lower()

    # 1. Alias translation for international brand names
    if clean_query in BRAND_ALIASES:
        clean_query = BRAND_ALIASES[clean_query]

    # 2. Case-insensitive database match
    match = df[
        df["Name"].str.contains(clean_query, case=False, na=False)
        | df["Contains"].str.contains(clean_query, case=False, na=False)
    ]
    if match.empty:
        return None

    # 3. Sort: Prioritize oral tablets/capsules over injections/IVs
    match = match.copy()
    match["is_injection"] = match["Name"].str.contains(
        "Injection|Infusion|IV", case=False, na=False
    )
    match["ingredient_count"] = match["Contains"].str.count(r"\+")
    sorted_matches = match.sort_values(by=["is_injection", "ingredient_count"])

    target = sorted_matches.iloc[0]["Name"]
    active = sorted_matches.iloc[0]["Contains"]
    subs = df[(df["Contains"] == active) & (df["Name"] != target)][
        "Name"
    ].unique()

    return {
        "searchQuery": searchTerm,
        "matchedMedicine": target,
        "activeIngredient": active,
        "substitutes": list(subs[:top_n]),
        "prescriptionRequired": True,
        "confidenceScore": 95 if len(subs) > 0 else 50,
        "genericMatchFound": len(subs) > 0,
    }


def runModelA(medName, active, subs):
    system_prompt = (
        "You are an expert AI clinical pharmacist. Given a queried medicine and its active ingredients, "
        "evaluate if generic alternatives are safe bio-equivalents. Output ONLY RAW JSON containing: "
        '1) "safety_approved" (boolean), 2) "safety_score" (0-100), 3) "dosage_instructions" (string), and 4) "key_warnings" (string). '
        "Do not use markdown. Do not add comments."
    )
    user_prompt = f"Medicine: {medName}\nActive: {active}\nSubs: {subs}\n"

    for attempt in range(3):
        try:
            resA = client.chat.completions.create(
                model=os.getenv("MODEL_SAFETY", "moonshotai/Kimi-K2.6"),
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

            data = json.loads(raw_text, strict=False)
            reqID = getattr(resA, "id", "gnk-req-safety")
            return data, reqID, None
        except Exception as e:
            if attempt == 2:
                return {
                    "safety_approved": False,
                    "safety_score": 0,
                    "dosage_instructions": "N/A",
                    "key_warnings": "Error",
                }, "err", str(e)
            time.sleep(2)


def runModelB(medName, location="Petaling Jaya, Selangor"):
    system_prompt = (
        f"You are a retail pharmaceutical inventory AI. Analyze stock risk and store availability for {medName}. "
        'Output ONLY RAW JSON containing: 1) "stock_risk" (\'Low\' | \'Medium\' | \'High\'), '
        '2) "nearest_chain_availability": array of store names like ["Watsons", "Guardian", "Local Pharmacy"], '
        'and 3) "estimated_in_stock_confidence" (0-100). Do not use markdown.'
    )
    user_prompt = f"Assess current market stock and retail store availability for {medName} in {location}."

    for attempt in range(3):
        try:
            resB = client.chat.completions.create(
                model=os.getenv("MODEL_SUPPLY", "deepseek-ai/DeepSeek-V4-Flash-0731"),
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

            data = json.loads(raw_text, strict=False)
            reqID = getattr(resB, "id", "gnk-req-supply")
            return data, reqID, None

        except Exception as e:
            if attempt == 2:
                return {
                    "stock_risk": "Unknown",
                    "nearest_chain_availability": [],
                    "estimated_in_stock_confidence": 0,
                }, "err", str(e)
            time.sleep(2)


def evaluateSingleMedi(searchTerm, top_n=5):
    lookup = findSubstitutes(searchTerm)
    if not lookup:
        return {
            "Search Query": searchTerm,
            "Matched Medicine": "Not Found",
            "Generic Match": "0 Subs",
            "Stock Risk": "N/A",
            "Stock Chains": "N/A",
            "Safety": "N/A",
            "Consensus Score": "0.0%",
            "Status": "Missing in database",
        }
    medName = lookup["matchedMedicine"]
    active = lookup["activeIngredient"]
    subs = lookup["substitutes"]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_a = executor.submit(runModelA, medName, active, subs)
        f_b = executor.submit(runModelB, medName)
        dataA, reqIDA, errA = f_a.result()
        dataB, reqIDB, errB = f_b.result()

    genericMatchPTS = 100 if lookup["genericMatchFound"] else 0
    stockScore = float(dataB.get("estimated_in_stock_confidence", 0))
    safetyScore = float(
        dataA.get("safety_score", 100 if dataA.get("safety_approved") else 0)
    )

    finalAvailabilityScore = (
        (genericMatchPTS * 0.4) + (stockScore * 0.4) + (safetyScore * 0.2)
    )
    stores = ", ".join(dataB.get("nearest_chain_availability", []))
    status = "OK" if not errA and not errB else f"API Error: {errA or errB}"

    return {
        "Search Query": searchTerm,
        "Matched Medicine": medName,
        "Generic Match": f"{len(subs)} found",
        "Stock Risk": dataB.get("stock_risk", "N/A"),
        "Stock Chains": stores if stores else "Local Pharmacy",
        "Safety": "Approved" if dataA.get("safety_approved") else "Review",
        "Consensus Score": f"{finalAvailabilityScore:.1f}%",
        "Status": status,
    }


if __name__ == "__main__":
    testList = [
        "Augmentin",
        "Paracetamol",
        "Metformin",
        "Amoxicillin",
        "Pantoprazole",
        "Atorvastatin",
        "Azithromycin",
        "Cetirizine",
        "Omeprazole",
        "Ibuprofen",
    ]

    print("\nRunning batch test...")
    results = []

    workers = int(os.getenv("MAX_WORKERS", 3))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futureToMed = {
            executor.submit(evaluateSingleMedi, med): med for med in testList
        }
        for idx, future in enumerate(
            concurrent.futures.as_completed(futureToMed), 1
        ):
            med = futureToMed[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[{idx}/10] Testing: {med.upper()}")
                print(f"Matched Medicine: {result['Matched Medicine']}")
                print(f"Generic Match: {result['Generic Match']}")
                print(f"Stock Risk: {result['Stock Risk']}")
                print(f"Availability: {result['Stock Chains']}")
                print(f"Safety: {result['Safety']}")
                print(f"Final Score: {result['Consensus Score']}")
                print(f"Status: {result['Status']}\n")
                time.sleep(1.0)
            except Exception as e:
                print(f"[{idx}/10] Error testing {med}: {e}\n")

        print("Batch Test Summary:")
        summary_df = pd.DataFrame(results)[
            ["Search Query", "Stock Risk", "Safety", "Consensus Score", "Status"]
        ]
        print(summary_df.to_string(index=False))
        print("\nBatch Test Complete")
