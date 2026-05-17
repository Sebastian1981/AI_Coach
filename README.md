# AI Coach — Personalized Training Advisor

> Science-backed, AI-powered training guidance tailored to your individual fitness data and goals.

---

## What Is This?

AI Coach is a personal training consultant powered by large language models (LLMs). It combines **your real training data** from a Garmin Forerunner 265 with the **current state of scientific knowledge** in endurance sports and strength training to deliver practical, personalized training advice.

The system is designed for **non-professional athletes** who want to:

- Train smarter, not just harder
- Avoid overtraining and reduce injury risk
- Get the maximum benefit from a minimal, sustainable training load
- Work toward specific performance goals

---

## Goals & Use Cases

You can define a concrete target and receive a tailored training plan:

| Goal | Example |
|------|---------|
| Endurance | Complete a half marathon or full marathon |
| Middle distance | Run a 5K or 10K at target pace |
| Speed | Maximize performance over 400m or 100m sprint |
| Fitness | General cardiovascular health and body composition |
| Strength | Improve gym performance alongside running/cycling |

For each goal, the system factors in your **current fitness level**, **recent training load**, **recovery state**, and **subjective well-being** to recommend a weekly or monthly training plan.

---

## Data Sources

### Personal Training Data (Garmin Forerunner 265)
- Activity files (`.fit` format) covering running, cycling, and gym sessions
- Heart rate, pace, power, cadence, GPS data
- HRV, VO2max estimates, sleep quality, and training load metrics
- Subjective notes: perceived effort, energy levels, how you felt

### Scientific Knowledge Base
- Peer-reviewed papers on endurance training periodization, polarized training, HIIT, and strength-endurance integration
- Research on overtraining syndrome, recovery markers, and injury prevention
- Evidence-based guidelines for amateur athletes from sources such as ACSM, NSCA, and leading sports science journals
- Papers are downloaded and indexed locally for offline use

---

## Technical Architecture

```
Garmin Forerunner 265
        │
        ▼
  .fit File Parser          ← fitparse / garminconnect
        │
        ▼
  Activity Processor        ← pandas, structured summaries
        │
        ▼
  Vector Store (RAG)        ← FAISS / ChromaDB
        │                   ← Scientific papers + personal data
        ▼
  MCP Server                ← Model Context Protocol
        │
        ▼
  LLM (e.g. GPT-4o)        ← OpenAI API / local model
        │
        ▼
  Coaching Response         ← Weekly/monthly training plan
```

### Key Technologies

| Component | Technology |
|-----------|-----------|
| Activity download | USB `.fit` export or `garminconnect` Python library |
| FIT file parsing | `fitparse` |
| Data processing | `pandas`, `numpy` |
| Scientific paper ingestion | `pypdf`, `unstructured` |
| Embeddings & RAG | `FAISS` or `ChromaDB` + `sentence-transformers` / OpenAI Embeddings |
| LLM interface | `openai` SDK or local model via `ollama` |
| Agent / tool use | MCP (Model Context Protocol) server |

---

## Guiding Principles

- **Minimal effective dose** — recommendations follow the principle of achieving the training stimulus with the least possible stress on the body
- **Overtraining prevention** — the system monitors cumulative load, HRV trends, and recovery indicators to flag when to back off
- **Evidence-based** — all advice is grounded in peer-reviewed sports science, not generic online advice
- **Individual-first** — your data always takes priority over generic templates

---

## Project Status

Work in progress — Proof of Concept phase

- [ ] FIT file download & parsing pipeline
- [ ] Activity feature extraction (load, intensity zones, recovery)
- [ ] Scientific paper ingestion & vectorization
- [ ] RAG pipeline connecting papers to personal context
- [ ] MCP server implementation
- [ ] LLM coaching prompt engineering
- [ ] Goal-setting interface & training plan output

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Sebastian1981/AI_Coach.git
cd AI_Coach

# 2. Install dependencies
pip install -r requirements.txt

# 3. Connect Garmin Forerunner 265 via USB
#    Copy .fit files from GARMIN/Activity/ to data/activities/

# 4. Run the activity parser
python src/parse_activities.py
```

---

## License

See [LICENSE](LICENSE).
