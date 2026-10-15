# Codveda Internship — Data Analytics Dashboard

A production-grade, multi-dataset analytics dashboard built with **Plotly Dash**.

[![Full Dash App](https://img.shields.io/badge/Full%20Dash%20App-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://codveda-internship-cwdk.onrender.com)
[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-chayma--saad.github.io-7de8c8?style=for-the-badge&logo=githubpages&logoColor=black)](https://chayma-saad.github.io/Codveda_internship/iris_dashboard.html)

> 🚀 **Full interactive Dash app (all 8 tasks, 4 datasets):** https://codveda-internship-cwdk.onrender.com — free tier, the first visit can take ~1 min while the server wakes up.
>
> 📊 **Static Iris dashboard:** https://chayma-saad.github.io/Codveda_internship/iris_dashboard.html

## Datasets Covered

| Dataset | Source | Tasks |
|---------|--------|-------|
| `iris.csv` | UCI / sklearn | L1-T1 Cleaning · L1-T2 EDA · L1-T3 Viz · L2-T3 Clustering · L3-T1 Classification |
| `stock_prices.csv` | Synthetic (realistic) | L2-T2 Time-Series Analysis |
| `house_prices.csv` | Synthetic (realistic) | L2-T1 Regression Analysis |
| `sentiment_dataset.csv` | Synthetic reviews | L3-T3 NLP Sentiment Analysis |

> **Note:** If you have the real Codveda datasets, just place them in the same folder as `app.py`
> with these exact filenames and the app will pick them up automatically.

## Setup

```bash
# 1. Clone / download the project folder
git clone https://github.com/chayma-saad/Codveda_internship.git
cd Codveda_internship

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download TextBlob corpora (first time only)
python -c "import nltk; nltk.download('punkt')"

# 5. Run
python app.py
```

Open **http://127.0.0.1:8050** in your browser.

## Project Structure

```
codveda-dashboard/
├── app.py                   ← main application
├── requirements.txt         ← all dependencies
├── README.md
├── iris.csv                 ← optional (auto-generated if missing)
├── stock_prices.csv         ← optional
├── house_prices.csv         ← optional
└── sentiment_dataset.csv    ← optional
```

## Deploy for Free (Render.com)

1. Push this folder to a **GitHub repo**
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:server`
5. Deploy → you get a live public URL for your LinkedIn post!

## Dashboard Sections

| Tab | Level | What it shows |
|-----|-------|---------------|
| L1 — Cleaning | Basic | Null audit, IQR outlier detection, dedup |
| L1 — EDA | Basic | Stats table, correlation heatmap, scatter matrix |
| L1 — Visualisation | Basic | Box plots, violin, bar, scatter — export-ready |
| L2 — Regression | Intermediate | House price prediction, residual plots, coefficients |
| L2 — Time Series | Intermediate | Candlestick, decomposition, MA smoothing, returns |
| L2 — Clustering | Intermediate | Elbow + silhouette, K-Means vs true labels, 3-D scatter |
| L3 — Classification | Advanced | 3 models, confusion matrix, feature importance, CV |
| L3 — NLP Sentiment | Advanced | TextBlob scoring, word frequencies, live analyser |

## LinkedIn Post Template

> Just completed the **Codveda Technology Data Analytics Internship** 🌿
>
> Built a full **8-section interactive dashboard** covering:
> - Data cleaning & EDA (Iris)
> - Linear regression on house prices
> - Time-series decomposition on stock data
> - K-Means clustering with silhouette analysis
> - Multi-model classification (93.3% accuracy)
> - NLP sentiment analysis with a live text analyser
>
> 👉 Live demo: [your-render-url]
> 📂 GitHub: [your-repo-link]
>
> #CodvedaJourney #CodvedaExperience #FutureWithCodveda
> #DataScience #MachineLearning #Python #Plotly #Dash
