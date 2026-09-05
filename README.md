# MUBA-HACKATHON-2026-medifind-ai-engine

**[🟢 LIVE DEMO: Test the MediFind AI Engine Here]**
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

⚙️ System Architecture
Frontend (UI/UX): Built with Streamlit for a highly responsive, interactive dashboard featuring dynamic search toggles, metric cards, and downloadable safety reports.

Backend Engine: Powered by Python & Pandas for lightning-fast, case-insensitive dataset querying and active ingredient mapping.

AI Orchestration: Implements asynchronous, concurrent API routing via concurrent.futures to ping multiple LLMs simultaneously without bottlenecking load times.

Resilience: Features custom Regex-based JSON extraction and automatic retry loops to safely handle LLM hallucinations and API rate limits.

✨ Key Features
Bi-Directional Search: Instantly query by Brand Name (e.g., Panadol) or Active Ingredient (e.g., Paracetamol).

Dual-Agent AI Consensus:

Model A (Moonshot Kimi-K2.6): Acts as a Clinical Pharmacist, scoring bio-equivalent safety and flagging contraindications.

Model B (DeepSeek-V4-Flash): Acts as a Supply Chain Analyst, projecting localized stock risks and mapping retail availability.

Cost Savings Alert: Automatically highlights when generic substitutes are found to help consumers save 50-75% on prescriptions.

Exportable Data: Generate and download a plain-text Pharmacist Safety Brief for offline viewing or medical consultations.

Enhance your repository's **README.md** by adding these production-ready Markdown sections covering installation, AI model routing, environment variables, and future roadmap.

Click the **Copy** button on any snippet below to paste it directly into your GitHub editor.

---

### Badges & Quick Start Header

*Place this directly below your main title to give reviewers instant links and build status.*

```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://muba-hackathon-2026-medifind-ai-engine-hykaugcpmym622wcraxvqj.streamlit.app/)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Hackathon](https://img.shields.io/badge/MUBA--Hackathon-2026-brightgreen)

---

```

---

### AI Model & Gonka Router Specification Table

*Place this under **System Architecture** to showcase your multi-model setup to hackathon judges.*

```markdown
## 🤖 Dual-Agent AI Orchestration (Gonka Router)

MediFind executes parallel, non-blocking LLM requests through **Gonka Router** using custom system prompts and strict JSON enforcement:

| Engine Component | Model | Provider / Endpoint | Key Output / Task |
| :--- | :--- | :--- | :--- |
| **Clinical Safety Engine** | `moonshotai/Kimi-K2.6` | Gonka Router API | Bio-equivalence evaluation, safety risk scoring (0-100), clinical contraindications, dosage guidance |
| **Supply Chain Engine** | `deepseek-ai/DeepSeek-V4-Flash-0731` | Gonka Router API | Regional pharmacy stock confidence, retail chain mapping (*Watsons, Guardian, Caring, BIG Pharmacy*), supply risk levels |

```

---

### Local Installation & Quick Start Guide

*Essential for developers or hackathon evaluators running your app locally.*

```markdown
## 🚀 Quick Start & Local Development

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/MUBA-HACKATHON-2026-medifind-ai-engine.git](https://github.com/your-username/MUBA-HACKATHON-2026-medifind-ai-engine.git)
cd MUBA-HACKATHON-2026-medifind-ai-engine

```

### Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
GONKA_API_KEY=your_gonka_api_key_here
GONKA_BASE_URL=[https://api.gonkarouter.io/v1](https://api.gonkarouter.io/v1)
MODEL_SAFETY=moonshotai/Kimi-K2.6
MODEL_SUPPLY=deepseek-ai/DeepSeek-V4-Flash-0731
DATASET_PATH=cleaned_medicines_final.csv

```

### 4. Run the Streamlit Application

```bash
streamlit run app.py

```

```

---

### Cost Savings Methodology
*Adds depth to the economic value pitch of your application.*

```markdown
## 💡 Cost Savings Engine Methodology

MediFind includes an integrated **Generic Price & Consumer Savings Calculator**:
* **Generic Discount Benchmark:** Uses an empirical **65% average market discount rate** when comparing branded pharmaceuticals to certified generic equivalents.
* **Projections Generated:** Real-time calculation of estimated generic price, monthly savings (RM), and projected 12-month annual consumer savings.

```

---

### Future Roadmap & Hackathon Acknowledgments

*Concludes the README with project vision and credits.*

```markdown
## 🗺️ Roadmap & Future Scope

- [ ] **OCR Prescription Scanner:** Upload image/PDF prescriptions to automatically query active ingredients and substitutes.
- [ ] **Real-Time Pharmacy API Sync:** Direct inventory integrations with retail pharmacy APIs for exact store stock counts.
- [ ] **Multi-Language Support:** Localized clinical safety briefs in Bahasa Malaysia, Mandarin, and English.

## ⚠️ Project Limitations

* **Non-Real-Time Inventory Data:** Retail stock availability and store mapping are generated using AI estimation models (DeepSeek-V4) based on regional distribution patterns. The app does not connect to live pharmacy POS or real-time warehouse inventory APIs.
* **Medical Disclaimer:** Clinical safety evaluations, bio-equivalence scores, and dosage guidance (Kimi-K2.6) are provided strictly for educational and informational purposes. They do not replace professional diagnosis, medical advice, or official consultation with a licensed pharmacist or physician.
* **Memory & Resource Caps:** When hosted on Streamlit Community Cloud, the application is constrained by a strict 1 GB RAM limit. Processing heavy concurrent queries or loading the full production dataset (~147k records) can cause container memory exhaustion (`Resource Limit Exceeded`).
* **Static Database Index:** Drug lookups rely on static offline datasets (`sample_medicines.csv` and `cleaned_medicines_final.csv.gz`). Drug listings, brand names, and active ingredients do not automatically synchronize with live regulatory databases (such as NPRA or FDA).
* **API Gateway Dependency:** Dual-model inference requires an active network connection and a valid `GONKA_API_KEY` targeting `api.gonkarouter.io`. If the gateway is unreachable or the key is omitted, the application falls back to offline UI demo mode.
* **Geocoding Rate Limits & Regional Scope:** Location reverse-geocoding relies on Nominatim (OpenStreetMap), which enforces a strict 1 request/second rate limit. Preset location options, state mappings, and currency calculations (MYR) are primarily configured for Malaysian geographies.

## 🏆 Acknowledgments

Built for the **MUBA Hackathon 2026**. Powered by [Gonka Router](https://gonkarouter.io/) for multi-model AI routing and infrastructure.

```

🏆 Acknowledgments & Credits
Hackathon: Built for MUBA Hackathon 2026.

AI Orchestration: Powered by Gonka Router for multi-model AI routing.

AI Development Assistant: Developed with Google Gemini as the primary AI coding assistant for code refactoring, fault-tolerant logic, and UI architecture.

📄 License
Distributed under the MIT License. See LICENSE for details.
