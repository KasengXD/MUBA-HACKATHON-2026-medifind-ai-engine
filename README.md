# 💊 MUBA-HACKATHON-2026-medifind-ai-engine

**[🟢 LIVE DEMO: Test the MediFind AI Engine Here](https://muba-hackathon-2026-medifind-ai-engine-hykaugcpmym622wcraxvqj.streamlit.app/)**

**MediFind AI** is a public medicine locator and generic substitution engine powered by **Gonka Router** (`gonkarouter.io`). It cross-verifies drug availability, bio-equivalent generic alternatives, and local retail pharmacy stock data via multi-model AI, outputting a unified 0–100% Consensus Score, clinical safety breakdown, cost savings analysis, and audit-ready Gonka Request IDs.

---

## ⚙️ System Architecture & Dual-Agent AI Orchestration

MediFind executes non-blocking, parallel LLM requests through **Gonka Router** using strict JSON enforcement, exponential backoff retries, and rate-limit staggering:

```text
                          ┌───────────────────────────┐
                          │   User Query / Location   │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │   RapidFuzz & Pandas Index  │
                         └──────────────┬──────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                │ ThreadPoolExecutor Concurrent Async Routing   │
                └──────────────┬─────────────────┬──────────────┘
                               │                 │
     (Clinical Safety)         │                 │         (Supply Chain)
┌──────────────────────────────▼───┐         ┌───▼─────────────────────────────┐
│ Gonka Gateway: Kimi-K2.6         │         │ Gonka Gateway: DeepSeek-V4      │
│ - Bio-Equivalence Approval       │         │ - Local Stock Confidence        │
│ - Dosage & Contraindications     │         │ - Nearby Retail Chain Mapping   │
└──────────────────────────────┬───┘         └───┬─────────────────────────────┘
                               │                 │
                               └────────┬────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │   Unified Consensus Score │
                          │   & UI Safety Brief Export│
                          └───────────────────────────┘

```

| Engine Component | Target Model | Gateway Endpoint | Key Task / Output |
| --- | --- | --- | --- |
| **Clinical Safety Engine** | `moonshotai/Kimi-K2.6` | `api.gonkarouter.io/v1` | Bio-equivalence verification, safety score (0-100), clinical contraindications, dosage guidance |
| **Supply Chain Engine** | `deepseek-ai/DeepSeek-V4-Flash-0731` | `api.gonkarouter.io/v1` | Regional stock availability confidence, retail chain mapping (*Watsons, Guardian, Caring, BIG Pharmacy*), supply risk scoring |

---

## 📐 Consensus Score Mathematical Methodology

The engine calculates a unified **Consensus Score** ($C$) on a 0–100% scale using a tri-factor weighted formula:

$$C = (0.40 \times G) + (0.40 \times S) + (0.20 \times K)$$

* $G = \text{Generic Availability Score}$ ($100\%$ if 1 or more generic alternatives are indexed, else $0\%$).
* $S = \text{Stock Confidence Score}$ ($0 - 100\%$ projected by DeepSeek-V4 based on regional supply chain models).
* $K = \text{Clinical Safety Score}$ ($0 - 100\%$ assigned by Kimi-K2.6 upon evaluating drug bio-equivalence).

---

## ✨ Key Features

* **Bi-Directional Brand & Active Search:** Query by popular brand names (*Panadol, Augmentin, Lipitor*) or active chemical formulations (*Paracetamol, Amoxicillin*).
* **Smart Brand Aliasing & Autocorrect:** Automatically resolves brand synonyms and leverages `RapidFuzz` fuzzy string matching for misspelled queries.
* **Variant Selector:** Automatically detects and lets users switch between different dosage strengths or product variants (e.g., *500mg vs 1000mg*).
* **3-Mode Location & GIS Engine:** Filter inventory by Malaysian State & City dropdowns, an interactive `Folium` map pin with `geopy` reverse-geocoding, or free-text location inputs.
* **Interactive Generic Substitutes Table:** View active ingredient matches, brand names, and bio-equivalence status in a structured dataset viewer.
* **Estimated Generic Cost Savings Calculator:** Dynamically computes estimated monthly and 12-month annual consumer savings (RM) based on an empirical **65% generic discount rate**.
* **Audit-Ready Safety Brief Export:** Download a plain-text Pharmacist Safety Brief complete with Gonka Request IDs (`req_id`) for offline medical consultation.
* **Medicine Reference Library:** A dedicated catalog tab with real-time text filtering across all indexed drug records.

---

## 📊 Dataset Setup & Provenance

This project indexes up to 147,852 pharmaceutical records:

* **Original Source:** [Kaggle - Medical Information Dataset](https://www.kaggle.com/datasets/imtkaggleteam/medical-information-dataset)
* **Processed Dataset:** `cleaned_medicines_final.csv.gz` (Cleaned & compressed via `data cleaning new.ipynb` for memory-efficient loading).
* **Lightweight Fallback:** `sample_medicines.csv` (~10k records) for fast local debugging.

---

## 🚀 Quick Start & Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MUBA-HACKATHON-2026-medifind-ai-engine.git
cd MUBA-HACKATHON-2026-medifind-ai-engine

```

### 2. Set Up Virtual Environment & Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
GONKA_API_KEY=your_gonka_api_key_here
GONKA_BASE_URL=https://api.gonkarouter.io/v1
MODEL_SAFETY=moonshotai/Kimi-K2.6
MODEL_SUPPLY=deepseek-ai/DeepSeek-V4-Flash-0731
DATASET_PATH=cleaned_medicines_final.csv.gz

```

### 4. Run the Application

```bash
streamlit run app.py

```

---

## ⚠️ Project Limitations

* **Non-Real-Time Inventory:** Retail stock availability and store mapping are generated via AI estimation models (DeepSeek-V4) based on regional distribution patterns and do not connect to live store POS systems.
* **Medical Disclaimer:** Clinical safety evaluations and dosage guidance (Kimi-K2.6) are provided strictly for educational and informational purposes. They do not replace professional diagnosis or consultation with a licensed healthcare provider.
* **Cloud Memory Constraints:** When hosted on free-tier Streamlit Community Cloud (~1 GB RAM limit), the app defaults to optimized compressed datasets (`.csv.gz`) or sample modes to prevent container memory exhaustion.
* **Geocoding Limits:** Location reverse-geocoding relies on Nominatim (OpenStreetMap), which enforces a strict 1 request/second rate limit. Preset location mappings are tuned for Malaysian geographies.

---

## 🗺️ Roadmap & Future Scope

* [ ] **OCR Prescription Scanner:** Upload image or PDF prescriptions to extract active ingredients automatically.
* [ ] **Direct Pharmacy POS Sync:** Integrate live inventory webhooks with major retail chains.
* [ ] **Multi-Language Support:** Generate safety briefs in Bahasa Malaysia, Mandarin, and Tamil.

---

## 🏆 Acknowledgments & Credits

* **Hackathon:** Built for the **MUBA Hackathon 2026**.
* **AI Orchestration:** Powered by [Gonka Router](https://gonkarouter.io/) for multi-model AI routing.
* **Development Partner:** Developed with **Google Gemini** as the primary AI collaborator for logic refactoring, fault-tolerant multithreading, and UI architecture.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
