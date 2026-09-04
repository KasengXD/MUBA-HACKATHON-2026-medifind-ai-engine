# MUBA-HACKATHON-2026-medifind-ai-engine

**[🟢 LIVE DEMO: Test the MediFind AI Engine Here]
(https://muba-hackathon-2026-medifind-ai-engine-hykaugcpmym622wcraxvqj.streamlit.app/)

MediFind AI is a public medicine locator powered by Gonka Router (gonkarouter.io). It cross-verifies drug availability, active generic alternatives, and local pharmacy stock data via multi-model AI, outputting a 0–100% Availability Confidence Score, safety breakdown, and transparent Gonka Request IDs.

## 📊 Dataset Setup & Provenance

This project utilizes a cleaned 147,852-row medicine dataset derived from an open-source medical catalog.

* **Original Dataset (Uncleaned):** [Kaggle - Medical Information Dataset](https://www.kaggle.com/datasets/imtkaggleteam/medical-information-dataset)
* **Processed Dataset (Cleaned):** Download [`cleaned_medicines_final.csv`](https://drive.google.com/file/d/1N3Sk5ebg-4-ZtpE0rcKKTpDV_hPY2g6Y/view?usp=sharing) *(Cleaned via `data cleaning new.ipynb`)*

**Setup Steps**

1. Download `cleaned_medicines_final.csv` from the Google Drive link above.
2. Place the downloaded file into the root directory of this repository.
3. Update your local `.env` file to reference the full dataset:
   ```env
   DATASET_PATH=cleaned_medicines_final.csv

## ⚙️ System Architecture

* **Frontend (UI/UX):** Built with **Streamlit** for a highly responsive, interactive dashboard featuring dynamic search toggles, metric cards, and downloadable safety reports.
* **Backend Engine:** Powered by **Python & Pandas** for lightning-fast, case-insensitive dataset querying and active ingredient mapping.
* **AI Orchestration:** Implements asynchronous, concurrent API routing via `concurrent.futures` to ping multiple LLMs simultaneously without bottlenecking load times.
* **Resilience:** Features custom Regex-based JSON extraction andautomatic retry loops to safely handle LLM hallucinations and API rate limits.

## ✨ Key Features

* **Bi-Directional Search:** Instanly query by Brand Name (e.g., *Panadol*) or Active Ingredient (e.g., *Paracetamol*).
* **Dual-Agent AI Consensus:** 
   * **Model A (Moonshot Kimi-K2.6):** Acts as a Clinical Pharmacist, scoring bio-equivalent safety and flagging contraindications.
   * **Model B (DeepSeek-V4-Flash):** Acts as a Supply Chain Analyst, projecting localized stock risks and mapping retail availability.
* **Cost Savings Alert:** Automatically highlights when generic substitutes are found to help consumers save 50-75% on prescriptions.
* **Exportable Data:** Generate and download a plain-text Pharmacist Safety Brief for offline viewing or medical consultations.