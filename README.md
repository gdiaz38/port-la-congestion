# 🚢 Port of LA / Long Beach — Live Congestion Tracker

A real-time vessel tracking and congestion prediction system for the Port of Los Angeles and Long Beach — the busiest port complex in the Western Hemisphere. Refreshed daily via live AIS vessel data with zero manual intervention.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-automated-2088FF?logo=githubactions)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📊 Live Dashboard

👉 **[View Live App](https://gdiaz38-port-la-congestion.streamlit.app)**

---

## Overview

Port congestion costs the US economy billions annually. This project ingests **real AIS vessel positions** from the LA/Long Beach port complex every day, scores each vessel through a trained Gradient Boosting model, and surfaces predicted truck turnaround times and congestion levels across all five terminal operators.

Key question it answers: *How congested is the port right now, and which terminals are under the most pressure?*

---

## Features

- **Live vessel map** — real-time AIS positions of all vessels in the LA/Long Beach bounding box
- **ML congestion scoring** — GBR model (R²=0.871, MAE=11.1 hrs) predicts truck turnaround per vessel
- **Anchorage detection** — vessels anchored and waiting are flagged as delayed
- **Terminal breakdown** — congestion levels across APM Terminals, TraPac, Yang Ming, Everport, China Shipping
- **30-day rolling history** — trend line showing how port congestion evolves over time
- **Daily automated refresh** via GitHub Actions cron — no manual work after deployment

---

## Data Sources

| Source | Data | Update Frequency |
|---|---|---|
| [AISstream.io](https://aisstream.io) | Live vessel positions, speed, vessel type | Daily (90-second burst) |
| Port of LA Annual TEU Stats | Historical throughput calibration | Used for model training |
| Port historical patterns | Seasonal congestion weights | Used for model training |

---

## Project Structure

```
port-la-congestion/
├── .github/
│   └── workflows/
│       └── refresh.yml           # Daily cron — AIS fetch, score, commit
├── app/
│   └── dashboard.py              # Streamlit dashboard
├── scripts/
│   ├── fetch_ais.py              # AISstream.io WebSocket collector
│   ├── predict.py                # Load model, score live vessels
│   ├── pipeline.py               # Orchestrates fetch → score → save
│   ├── train_and_save.py         # One-time model training → delay_model.pkl
│   └── build_data_original.py    # Historical synthetic data generator
├── models/
│   └── delay_model.pkl           # Trained GBR model + label encoders
├── data/
│   ├── vessels_live.csv          # Latest AIS snapshot (committed by Actions)
│   └── vessels_history.csv       # Rolling 30-day vessel history
└── requirements.txt
```

---

## How It Works

```
GitHub Actions (cron: 7am Pacific daily)
        ↓
fetch_ais.py opens WebSocket to aisstream.io
Collects all vessels in LA/Long Beach bounding box for 90 seconds
        ↓
predict.py loads delay_model.pkl
Scores each vessel → predicted truck turnaround (hrs) + congestion level
        ↓
Writes vessels_live.csv + appends vessels_history.csv
Commits → pushes to main
        ↓
Streamlit Community Cloud detects push → auto-redeploys
```

---

## ML Model

Trained on 60,153 calibrated vessel arrival records (2019–2024) anchored to real Port of LA annual TEU statistics.

| Metric | Value |
|---|---|
| Algorithm | Gradient Boosting Regressor |
| Target | Truck turnaround time (hours) |
| R² Score | 0.871 |
| MAE | 11.1 hours |
| Top predictor | Crisis/disruption events (83.7% importance) |
| Peak season | August – October |

**Congestion tiers:**

| Level | Predicted Turnaround |
|---|---|
| 🟢 Low | < 80 hrs |
| 🟡 Moderate | 80–100 hrs |
| 🟠 High | 100–130 hrs |
| 🔴 Critical | > 130 hrs |

---

## Dashboard Sections

**KPI Row** — vessels tracked, anchored/delayed count, avg predicted turnaround, dominant congestion level

**Live Vessel Map** — real AIS positions colored by congestion level; hover for vessel name, speed, terminal, predicted turnaround

**Congestion & Vessel Type** — breakdown by tier and vessel classification

**Terminal Comparison** — avg predicted turnaround per terminal operator

**Historical Trend** — 30-day rolling daily average turnaround

**Vessel Table** — searchable by name, filterable by delayed status and congestion level

---

## Local Setup

### 1. Clone and create environment

```bash
git clone https://github.com/gdiaz38/port-la-congestion
cd port-la-congestion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get free AIS API key

Register at [aisstream.io](https://aisstream.io) → Dashboard → copy API key.

### 3. Create `.env`

```bash
AIS_API_KEY=your_key_here
```

### 4. Train model (one-time)

```bash
python3 scripts/build_data_original.py
python3 scripts/train_and_save.py
```

### 5. Run pipeline and launch dashboard

```bash
python3 scripts/pipeline.py
streamlit run app/dashboard.py
```

---

## Deployment

### GitHub Actions

Secret required: `AIS_API_KEY`

```bash
gh secret set AIS_API_KEY
```

Workflow runs daily at 7am Pacific, commits refreshed CSVs, pushes to main.

### Streamlit Community Cloud

1. Connect repo at [share.streamlit.io](https://share.streamlit.io)
2. Main file: `app/dashboard.py`
3. Add `AIS_API_KEY` under Advanced Settings → Secrets
4. Deploy

---

## Tech Stack

`Python 3.11` · `Streamlit` · `Plotly` · `Pandas` · `Scikit-learn` · `DuckDB` · `websockets` · `GitHub Actions` · `AISstream.io`

---

## Affiliation

University of California, Riverside — MS in Engineering Management
Part of a portfolio of 10 live data science projects spanning computer vision, NLP, supply chain, and healthcare ML.

---

## License

MIT
