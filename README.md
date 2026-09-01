# MUBA-HACKATHON-2026-medifind-ai-engine
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
