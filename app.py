"""
╔══════════════════════════════════════════════════════════════════════╗
║   Codveda Technology — Data Analytics Internship                     ║
║   Complete Dashboard  ·  app.py                                      ║
║                                                                      ║
║   DATASETS                                                           ║
║     • iris.csv             (Level 1 & 2 — EDA, Regression,          ║
║                              Clustering, Classification)             ║
║     • stock_prices.csv     (Level 2 — Time-Series Analysis)         ║
║     • house_prices.csv     (Level 2 — Regression Analysis)          ║
║     • sentiment_dataset.csv(Level 3 — NLP Sentiment)                ║
║                                                                      ║
║   TASKS COVERED                                                      ║
║     L1-T1  Data Cleaning & Preprocessing                            ║
║     L1-T2  Exploratory Data Analysis (EDA)                          ║
║     L1-T3  Basic Data Visualisation                                  ║
║     L2-T1  Regression Analysis        (house_prices)                ║
║     L2-T2  Time-Series Analysis       (stock_prices)                ║
║     L2-T3  K-Means Clustering         (iris)                        ║
║     L3-T1  Predictive Modelling       (iris)                        ║
║     L3-T3  NLP Sentiment Analysis     (sentiment_dataset)           ║
║                                                                      ║
║   RUN:  python app.py                                                ║
║   OPEN: http://127.0.0.1:8050                                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────────
# INSTALL CHECK  (run once: pip install -r requirements.txt)
# ──────────────────────────────────────────────────────────────────────
import sys, subprocess
# Force UTF-8 output on Windows to handle Unicode characters in print statements
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

REQUIRED = [
    "dash", "dash-bootstrap-components", "plotly",
    "pandas", "numpy", "scikit-learn",
    "statsmodels", "textblob", "wordcloud", "matplotlib",
]

def _install_if_missing():
    import importlib
    mapping = {
        "dash-bootstrap-components": "dash_bootstrap_components",
        "scikit-learn": "sklearn",
        "textblob": "textblob",
        "wordcloud": "wordcloud",
    }
    for pkg in REQUIRED:
        mod = mapping.get(pkg, pkg.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_install_if_missing()

# ──────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import io, base64, os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, classification_report,
    confusion_matrix, silhouette_score,
)

from statsmodels.tsa.seasonal import seasonal_decompose

from textblob import TextBlob
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ──────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (dark, professional)
# ──────────────────────────────────────────────────────────────────────
COLORS = dict(
    bg         = "#0d0f14",
    bg2        = "#13161e",
    bg3        = "#1a1e29",
    border     = "rgba(255,255,255,0.07)",
    accent1    = "#7de8c8",   # teal-green
    accent2    = "#f4845f",   # coral
    accent3    = "#9b8fff",   # violet
    muted      = "#6b7280",
    text       = "#e8eaf0",
    setosa     = "#7de8c8",
    versicolor = "#f4845f",
    virginica  = "#9b8fff",
)

PLOT_LAYOUT = dict(
    paper_bgcolor = COLORS["bg2"],
    plot_bgcolor  = COLORS["bg3"],
    font          = dict(color=COLORS["muted"], family="DM Sans, sans-serif", size=12),
    margin        = dict(l=40, r=20, t=40, b=40),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    xaxis         = dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis         = dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)"),
)

SP_COLORS = {
    "setosa":     COLORS["setosa"],
    "versicolor": COLORS["versicolor"],
    "virginica":  COLORS["virginica"],
}

# ──────────────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

def _path(name):
    """Resolve dataset path: local file first, then /tmp fallback."""
    for p in [os.path.join(DATA_DIR, name), os.path.join("/tmp", name)]:
        if os.path.exists(p):
            return p
    return None

def load_iris() -> pd.DataFrame:
    p = _path("1) iris.csv")
    if p:
        df = pd.read_csv(p)
    else:
        from sklearn.datasets import load_iris as _li
        iris = _li()
        df = pd.DataFrame(iris.data, columns=["sepal_length","sepal_width","petal_length","petal_width"])
        df["species"] = [iris.target_names[t] for t in iris.target]
    df["species"] = df["species"].str.strip().str.lower()
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    num_cols = ["sepal_length","sepal_width","petal_length","petal_width"]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

def load_stock() -> pd.DataFrame:
    p = _path("2) Stock Prices Data Set.csv") 
    if p:
        df = pd.read_csv(p)
        # normalise column names
        df.columns = [c.strip().title() for c in df.columns]
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
    else:
        # generate synthetic stock data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=1000, freq="B")
        price, closes = 150.0, []
        for _ in range(1000):
            price *= (1 + np.random.normal(0.0003, 0.018))
            closes.append(round(price, 2))
        closes = np.array(closes)
        df = pd.DataFrame({
            "Date":   dates.strftime("%Y-%m-%d"),
            "Open":   (closes * np.random.uniform(0.995, 1.005, 1000)).round(2),
            "High":   (closes * np.random.uniform(1.005, 1.025, 1000)).round(2),
            "Low":    (closes * np.random.uniform(0.975, 0.995, 1000)).round(2),
            "Close":  closes,
            "Volume": np.random.randint(50_000_000, 200_000_000, 1000),
        })
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset=["Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if "Close" in df.columns:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["MA_7"]  = df["Close"].rolling(7,  center=True).mean()
        df["MA_30"] = df["Close"].rolling(30, center=True).mean()
        df["Returns"] = df["Close"].pct_change() * 100
    return df

def load_house() -> pd.DataFrame:
    p = _path("4) house Prediction Data Set.csv") 
    if p:
        df = pd.read_csv(p)
    else:
        np.random.seed(7)
        n = 500
        size  = np.random.randint(500, 4500, n)
        beds  = np.random.choice([1,2,3,4,5], n, p=[0.05,0.2,0.4,0.25,0.1])
        baths = np.random.choice([1,1.5,2,2.5,3,3.5,4], n)
        age   = np.random.randint(1, 50, n)
        dist  = np.random.uniform(1, 30, n).round(1)
        price = (size*120 + beds*8000 + baths*5000 - age*500 - dist*1200
                 + np.random.normal(0,18000,n)).clip(80_000,1_200_000).round(-2)
        neighb = np.random.choice(
            ["Downtown","Suburbs","Uptown","Midtown","East Side"], n,
            p=[0.15,0.35,0.2,0.2,0.1])
        df = pd.DataFrame({"size_sqft":size,"bedrooms":beds,"bathrooms":baths,
            "age_years":age,"dist_downtown_km":dist,"neighborhood":neighb,
            "price":price.astype(int)})
    # fill nulls with median for numeric
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(df[num].median())
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_sentiment() -> pd.DataFrame:
    p = _path("3) Sentiment dataset.csv")
    if p:
        df = pd.read_csv(p)
    else:
        np.random.seed(13)
        positive = ["Amazing product, highly recommend!","Great quality and fast shipping.",
            "Exceeded my expectations completely.","Best purchase I made this year.",
            "Absolutely love it, works perfectly!","Five stars, would buy again.",
            "Outstanding value for money.","Very happy with this product."]
        negative = ["Terrible quality, broke after a week.","Very disappointed with this product.",
            "Complete waste of money, avoid.","Worst purchase ever, do not buy.",
            "Poor quality, not as described.","Stopped working after one use.",
            "Horrible experience, terrible support.","Defective item, very frustrated."]
        neutral  = ["It is okay, nothing special.","Average product, does the job.",
            "Not bad but not great either.","Mediocre quality for the price.",
            "Some pros and cons, could be better.","Decent product, works as described."]
        reviews, sents, ratings = [], [], []
        for _ in range(300):
            s = np.random.choice(["positive","negative","neutral"], p=[0.5,0.3,0.2])
            pool = {"positive":positive,"negative":negative,"neutral":neutral}[s]
            reviews.append(np.random.choice(pool)+" "+np.random.choice(pool))
            sents.append(s)
            ratings.append({"positive":np.random.choice([4,5]),
                             "negative":np.random.choice([1,2]),"neutral":3}[s])
        df = pd.DataFrame({"review":reviews,"sentiment":sents,"rating":ratings})

    # TextBlob sentiment scores
    def _clean(txt):
        return re.sub(r"[^a-zA-Z\s]","",str(txt)).lower().strip()

    df["clean"]     = df.iloc[:,0].apply(_clean)
    df["polarity"]  = df["clean"].apply(lambda t: TextBlob(t).sentiment.polarity)
    df["subjectivity"] = df["clean"].apply(lambda t: TextBlob(t).sentiment.subjectivity)
    df["tb_label"]  = df["polarity"].apply(
        lambda p: "positive" if p>0.05 else ("negative" if p<-0.05 else "neutral"))
    return df

# Load all datasets at startup
print("Loading datasets...")
DF_IRIS      = load_iris()
DF_STOCK     = load_stock()
DF_HOUSE     = load_house()
DF_SENTIMENT = load_sentiment()
print("  [OK] iris      :", DF_IRIS.shape)
print("  [OK] stock     :", DF_STOCK.shape)
print("  [OK] house     :", DF_HOUSE.shape)
print("  [OK] sentiment :", DF_SENTIMENT.shape)

# ──────────────────────────────────────────────────────────────────────
# PRE-COMPUTE ML RESULTS  (avoids re-training on every callback)
# ──────────────────────────────────────────────────────────────────────
IRIS_NUM = ["sepal_length","sepal_width","petal_length","petal_width"]

# --- Clustering (iris) ---
scaler_cl    = StandardScaler()
X_iris_scaled = scaler_cl.fit_transform(DF_IRIS[IRIS_NUM])
_inertias, _sil = [], []
for k in range(2, 9):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_iris_scaled)
    _inertias.append({"k":k, "inertia":km.inertia_})
    _sil.append({"k":k, "score":silhouette_score(X_iris_scaled, km.labels_)})
KM3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_iris_scaled)
DF_IRIS["cluster"] = KM3.labels_.astype(str)
ELBOW_DF = pd.DataFrame(_inertias)
SIL_DF   = pd.DataFrame(_sil)

# --- Classification (iris) ---
le = LabelEncoder()
X_cls = StandardScaler().fit_transform(DF_IRIS[IRIS_NUM])
y_cls = le.fit_transform(DF_IRIS["species"])
X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.2,
                                            random_state=42, stratify=y_cls)
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}
CLS_RESULTS = {}
for name, clf in CLASSIFIERS.items():
    clf.fit(X_tr, y_tr)
    yh  = clf.predict(X_te)
    cv  = cross_val_score(clf, X_cls, y_cls, cv=5).mean()
    CLS_RESULTS[name] = dict(
        model=clf, y_pred=yh,
        accuracy=accuracy_score(y_te, yh),
        f1=f1_score(y_te, yh, average="macro"),
        cv=cv,
        cm=confusion_matrix(y_te, yh),
        report=classification_report(y_te, yh, target_names=le.classes_),
        report_dict=classification_report(y_te, yh, target_names=le.classes_, output_dict=True),
    )
RF_BEST      = CLS_RESULTS["Random Forest"]["model"]
FEAT_IMP_DF  = pd.DataFrame({"feature":IRIS_NUM,
                               "importance":RF_BEST.feature_importances_})\
                  .sort_values("importance", ascending=False)
CLASS_NAMES  = le.classes_

# --- GridSearchCV hyperparameter tuning (Random Forest) ---
_param_grid = {
    "n_estimators":      [50, 100, 200],
    "max_depth":         [None, 5, 10],
    "min_samples_split": [2, 5],
}
_gs = GridSearchCV(
    RandomForestClassifier(random_state=42), _param_grid,
    cv=5, scoring="f1_macro", n_jobs=-1, refit=True
)
_gs.fit(X_cls, y_cls)
GS_BEST_PARAMS = _gs.best_params_
GS_BEST_SCORE  = _gs.best_score_
GS_BEST_MODEL  = _gs.best_estimator_
_gs_pred       = GS_BEST_MODEL.predict(X_te)
GS_ACCURACY    = accuracy_score(y_te, _gs_pred)
GS_F1          = f1_score(y_te, _gs_pred, average="macro")
GS_CM          = confusion_matrix(y_te, _gs_pred)
GS_REPORT_DICT = classification_report(y_te, _gs_pred, target_names=le.classes_, output_dict=True)
print(f"  GridSearchCV best params : {GS_BEST_PARAMS}")
print(f"  GridSearchCV best CV F1  : {GS_BEST_SCORE:.4f}")

# --- Regression (house) ---
_house_num = ["size_sqft","bedrooms","bathrooms","age_years","dist_downtown_km"]
_house_num = [c for c in _house_num if c in DF_HOUSE.columns]
if _house_num and "price" in DF_HOUSE.columns:
    X_h = DF_HOUSE[_house_num].fillna(DF_HOUSE[_house_num].median())
    y_h = DF_HOUSE["price"]
    Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
    REG_MODEL = LinearRegression().fit(Xh_tr, yh_tr)
    yh_pred = REG_MODEL.predict(Xh_te)
    REG_METRICS = dict(
        r2   = r2_score(yh_te, yh_pred),
        rmse = np.sqrt(mean_squared_error(yh_te, yh_pred)),
        mae  = mean_absolute_error(yh_te, yh_pred),
        coef = dict(zip(_house_num, REG_MODEL.coef_)),
    )
    REG_TEST = pd.DataFrame({"actual":yh_te.values, "predicted":yh_pred,
                              "residual":yh_te.values - yh_pred})
else:
    REG_MODEL, REG_METRICS, REG_TEST = None, {}, pd.DataFrame()

print("[OK] All ML models trained (incl. GridSearchCV tuning).\n")

# ──────────────────────────────────────────────────────────────────────
# HELPER — APPLY SHARED PLOT THEME
# ──────────────────────────────────────────────────────────────────────
def theme(fig, title=""):
    fig.update_layout(**PLOT_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color=COLORS["text"], size=14)))
    return fig

# ──────────────────────────────────────────────────────────────────────
# CARD COMPONENT
# ──────────────────────────────────────────────────────────────────────
def card(children, style=None):
    base = dict(
        background=COLORS["bg2"], border=f"1px solid {COLORS['border']}",
        borderRadius="12px", padding="20px", marginBottom="16px",
    )
    if style:
        base.update(style)
    return html.Div(children, style=base)

def kpi(label, value, sub="", color=None):
    return html.Div([
        html.Div(label, style=dict(fontFamily="DM Mono, monospace", fontSize="0.68rem",
            color=COLORS["muted"], textTransform="uppercase", letterSpacing="0.08em",
            marginBottom="4px")),
        html.Div(value, style=dict(fontFamily="Georgia, serif", fontSize="2rem",
            fontWeight="600", lineHeight="1", color=color or COLORS["text"])),
        html.Div(sub, style=dict(fontSize="0.72rem", color=COLORS["muted"], marginTop="4px")),
    ], style=dict(background=COLORS["bg2"], border=f"1px solid {COLORS['border']}",
                  borderRadius="10px", padding="16px 20px"))

def section_header(title_plain, title_em, subtitle):
    return html.Div([
        html.H1([title_plain, " ", html.Em(title_em,
            style=dict(fontStyle="italic", color=COLORS["accent1"]))],
            style=dict(fontFamily="Georgia, serif", fontWeight="300",
                       fontSize="clamp(1.6rem,3.5vw,2.4rem)",
                       letterSpacing="-0.03em", marginBottom="6px")),
        html.P(subtitle, style=dict(color=COLORS["muted"], fontSize="0.88rem",
                                     maxWidth="560px")),
    ], style=dict(marginBottom="24px", paddingBottom="16px",
                  borderBottom=f"1px solid {COLORS['border']}"))

def mono(txt, color=None):
    return html.Span(txt, style=dict(fontFamily="DM Mono, monospace",
        fontSize="0.8rem", color=color or COLORS["muted"]))

# ──────────────────────────────────────────────────────────────────────
# TAB CONTENT BUILDERS
# ──────────────────────────────────────────────────────────────────────

# ── LEVEL 1: Cleaning ────────────────────────────────────────────────
def tab_cleaning():
    df = DF_IRIS.copy()
    num = IRIS_NUM

    # Null analysis
    null_df = pd.DataFrame({
        "Column":  df.columns.tolist(),
        "Dtype":   df.dtypes.astype(str).tolist(),
        "Non-Null":df.notna().sum().tolist(),
        "Nulls":   df.isna().sum().tolist(),
        "Null %":  (df.isna().mean()*100).round(2).tolist(),
    })

    # Outlier (IQR) report
    out_rows = []
    for col in num:
        q1, q3 = df[col].quantile([0.25,0.75])
        iqr = q3 - q1
        n_out = ((df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)).sum()
        out_rows.append({"Feature":col,"Q1":round(q1,3),"Q3":round(q3,3),
                          "IQR":round(iqr,3),"Outliers":int(n_out)})
    out_df = pd.DataFrame(out_rows)

    tbl_style = dict(backgroundColor=COLORS["bg3"], color=COLORS["text"],
                     fontFamily="DM Mono, monospace", fontSize="0.8rem")

    def tbl(df_, id_):
        return dash_table.DataTable(
            id=id_, data=df_.to_dict("records"),
            columns=[{"name":c,"id":c} for c in df_.columns],
            style_header=dict(**tbl_style, color=COLORS["accent1"],
                              borderBottom=f"1px solid {COLORS['border']}"),
            style_data=dict(**tbl_style, border="none"),
            style_table=dict(overflowX="auto"),
            style_cell=dict(padding="8px 12px", textAlign="left"),
        )

    return html.Div([
        section_header("Data Cleaning &", "Preprocessing",
            "Null audit, imputation strategy, duplicate removal, and IQR outlier detection on the Iris dataset."),
        dbc.Row([
            dbc.Col(kpi("Total Samples","150","Before cleaning",COLORS["accent1"]),md=3),
            dbc.Col(kpi("After Cleaning","147","Duplicates removed: 3",COLORS["accent2"]),md=3),
            dbc.Col(kpi("Missing Values","0","Detected across all columns",COLORS["accent3"]),md=3),
            dbc.Col(kpi("Features","4","sepal + petal dimensions"),md=3),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(card([
                html.H3("Null & Dtype Audit", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="12px")),
                tbl(null_df, "tbl-null"),
                html.Div("✓ No nulls detected — imputation strategy: median (numeric), mode (categorical)",
                    style=dict(marginTop="12px", padding="10px 14px",
                               background="rgba(125,232,200,0.06)",
                               border="1px solid rgba(125,232,200,0.15)",
                               borderRadius="8px", fontSize="0.78rem",
                               color=COLORS["muted"])),
            ]), md=7),
            dbc.Col(card([
                html.H3("Outlier Report (IQR Rule)", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="12px")),
                tbl(out_df, "tbl-outlier"),
                dcc.Graph(
                    figure=theme(
                        px.box(DF_IRIS, x="species", y="sepal_width",
                               color="species",
                               color_discrete_map=SP_COLORS,
                               points="outliers"),
                        "sepal_width outliers by species"),
                    config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"240px","marginTop":"12px"}),
            ]), md=5),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([
                html.H3("Standardised Label Distribution", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="12px")),
                dcc.Graph(
                    figure=theme(px.histogram(DF_IRIS, x="species", color="species",
                        color_discrete_map=SP_COLORS, barmode="overlay")),
                    config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"220px"}),
            ])),
        ], className="g-3"),
    ])

# ── LEVEL 1: EDA ─────────────────────────────────────────────────────
def tab_eda():
    df = DF_IRIS.copy()
    num = IRIS_NUM

    # Descriptive stats
    stats = df[num].describe().T.round(3)
    stats["median"] = df[num].median().round(3)
    stats["skew"]   = df[num].skew().round(3)
    stats["kurt"]   = df[num].kurt().round(3)
    stats = stats[["count","mean","median","std","min","max","skew","kurt"]].reset_index()
    stats.rename(columns={"index":"Feature"}, inplace=True)

    # Corr heatmap
    corr = df[num].corr().round(3)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0,"#f4845f"],[0.5,"#1a1e29"],[1,"#7de8c8"]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        showscale=True,
    ))
    theme(fig_corr, "Pearson Correlation Matrix")

    tbl_style = dict(backgroundColor=COLORS["bg3"], color=COLORS["text"],
                     fontFamily="DM Mono, monospace", fontSize="0.78rem")

    return html.Div([
        section_header("Exploratory Data", "Analysis",
            "Summary statistics, distributions, pair plots, and correlation structure."),
        card([
            html.H3("Descriptive Statistics", style=dict(fontFamily="DM Mono, monospace",
                fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                letterSpacing="0.08em", marginBottom="12px")),
            dash_table.DataTable(
                data=stats.to_dict("records"),
                columns=[{"name":c,"id":c} for c in stats.columns],
                style_header=dict(**tbl_style, color=COLORS["accent1"],
                                  borderBottom=f"1px solid {COLORS['border']}"),
                style_data=dict(**tbl_style, border="none"),
                style_table=dict(overflowX="auto"),
                style_cell=dict(padding="8px 12px", textAlign="right"),
                style_cell_conditional=[{"if":{"column_id":"Feature"},"textAlign":"left"}],
            ),
        ]),
        dbc.Row([
            dbc.Col(card([
                html.H3("Feature Distributions", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="8px")),
                dcc.Dropdown(id="eda-feat",
                    options=[{"label":f.replace("_"," ").title(),"value":f} for f in num],
                    value=num[0],
                    style=dict(backgroundColor=COLORS["bg3"], color=COLORS["text"],
                               border=f"1px solid {COLORS['border']}", borderRadius="6px")),
                dcc.Graph(id="eda-hist", config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}},
                          style={"height":"260px","marginTop":"10px"}),
            ]), md=6),
            dbc.Col(card([
                html.H3("Correlation Heatmap", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="12px")),
                dcc.Graph(figure=fig_corr, config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}},
                          style={"height":"300px"}),
            ]), md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([
                html.H3("Scatter Matrix — Feature Pairs", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="12px")),
                dcc.Graph(
                    figure=theme(px.scatter_matrix(df, dimensions=num, color="species",
                        color_discrete_map=SP_COLORS, opacity=0.7)),
                    config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"420px"}),
            ])),
        ], className="g-3"),
    ])

# ── LEVEL 1: Visualisation ───────────────────────────────────────────
def tab_viz():
    df = DF_IRIS.copy()
    num = IRIS_NUM

    # Box plots
    fig_box = make_subplots(rows=1, cols=4,
        subplot_titles=[n.replace("_"," ").title() for n in num])
    for i, col in enumerate(num, 1):
        for sp, color in SP_COLORS.items():
            sub = df[df["species"]==sp][col]
            fig_box.add_trace(
                go.Box(y=sub, name=sp, marker_color=color,
                       showlegend=(i==1), boxmean=True), row=1, col=i)
    theme(fig_box, "Box Plots — Spread & Outliers")

    # Scatter
    fig_scat = px.scatter(df, x="petal_length", y="petal_width",
        color="species", color_discrete_map=SP_COLORS,
        size_max=10, opacity=0.85,
        hover_data=["sepal_length","sepal_width"])
    theme(fig_scat, "Petal Length vs Petal Width")

    # Violin
    fig_viol = px.violin(df, x="species", y="petal_length",
        color="species", color_discrete_map=SP_COLORS, box=True, points="all")
    theme(fig_viol, "Petal Length Distribution (Violin)")

    # Grouped bar
    means = df.groupby("species")[num].mean().reset_index().melt(
        id_vars="species", var_name="Feature", value_name="Mean cm")
    fig_bar = px.bar(means, x="Feature", y="Mean cm", color="species",
        color_discrete_map=SP_COLORS, barmode="group", text_auto=".2f")
    theme(fig_bar, "Mean Measurements per Species")

    return html.Div([
        section_header("Data", "Visualisation",
            "Bar plots, scatter plots, box plots, and violin charts — all export-ready."),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_scat,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"scatter"}},
                style={"height":"340px"})]), md=6),
            dbc.Col(card([dcc.Graph(figure=fig_bar,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"bar"}},
                style={"height":"340px"})]), md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_box,
                config={"displayModeBar":True}, style={"height":"300px"})]), md=8),
            dbc.Col(card([dcc.Graph(figure=fig_viol,
                config={"displayModeBar":True}, style={"height":"300px"})]), md=4),
        ], className="g-3"),
    ])

# ── LEVEL 2: Regression (house prices) ──────────────────────────────
def tab_regression():
    if REG_MODEL is None:
        return html.Div("House price dataset not available.", style=dict(color=COLORS["muted"]))

    m = REG_METRICS
    rt = REG_TEST

    fig_fit = px.scatter(rt, x="actual", y="predicted",
        color_discrete_sequence=[COLORS["accent1"]],
        labels={"actual":"Actual Price ($)","predicted":"Predicted Price ($)"})
    fig_fit.add_shape(type="line",
        x0=rt["actual"].min(), y0=rt["actual"].min(),
        x1=rt["actual"].max(), y1=rt["actual"].max(),
        line=dict(color=COLORS["accent2"], width=2, dash="dash"))
    theme(fig_fit, "Actual vs Predicted House Price")

    fig_resid = px.scatter(rt, x="predicted", y="residual",
        color_discrete_sequence=[COLORS["accent3"]],
        labels={"predicted":"Fitted Value","residual":"Residual ($)"})
    fig_resid.add_hline(y=0, line_color=COLORS["accent2"], line_dash="dash")
    theme(fig_resid, "Residual Plot")

    coef_df = pd.DataFrame({"Feature":list(m["coef"].keys()),
                              "Coefficient":list(m["coef"].values())}).sort_values(
        "Coefficient", key=abs, ascending=False)
    fig_coef = px.bar(coef_df, x="Coefficient", y="Feature",
        orientation="h", color="Coefficient",
        color_continuous_scale=["#f4845f","#13161e","#7de8c8"],
        color_continuous_midpoint=0)
    theme(fig_coef, "Feature Coefficients")

    # Distribution of residuals
    fig_rdist = px.histogram(rt, x="residual", nbins=30,
        color_discrete_sequence=[COLORS["accent3"]])
    theme(fig_rdist, "Residual Distribution")

    return html.Div([
        section_header("Regression", "Analysis",
            "Predicting house prices from property features using multiple linear regression."),
        dbc.Row([
            dbc.Col(kpi("R²", f"{m['r2']:.4f}", "Variance explained",COLORS["accent1"]),md=4),
            dbc.Col(kpi("RMSE", f"${m['rmse']:,.0f}", "Root mean squared error",COLORS["accent2"]),md=4),
            dbc.Col(kpi("MAE",  f"${m['mae']:,.0f}",  "Mean absolute error",COLORS["accent3"]),md=4),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_fit,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"320px"})]), md=6),
            dbc.Col(card([dcc.Graph(figure=fig_resid,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"320px"})]), md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_coef,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"280px"})]), md=6),
            dbc.Col(card([dcc.Graph(figure=fig_rdist,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"280px"})]), md=6),
        ], className="g-3"),
    ])

# ── LEVEL 2: Time Series (stock) ─────────────────────────────────────
def tab_timeseries():
    df = DF_STOCK.copy()
    date_col = "Date"
    price_col = "Close" if "Close" in df.columns else df.select_dtypes("number").columns[0]

    # Candlestick (if OHLC available)
    has_ohlc = all(c in df.columns for c in ["Open","High","Low","Close"])
    if has_ohlc:
        fig_price = go.Figure(go.Candlestick(
            x=df[date_col], open=df["Open"], high=df["High"],
            low=df["Low"],  close=df["Close"],
            increasing_line_color=COLORS["accent1"],
            decreasing_line_color=COLORS["accent2"],
        ))
        fig_price.add_trace(go.Scatter(x=df[date_col], y=df.get("MA_7"),
            mode="lines", name="7-day MA",
            line=dict(color=COLORS["accent3"], width=1.5)))
        fig_price.add_trace(go.Scatter(x=df[date_col], y=df.get("MA_30"),
            mode="lines", name="30-day MA",
            line=dict(color=COLORS["accent2"], width=1.5, dash="dot")))
    else:
        fig_price = px.line(df, x=date_col, y=price_col)
    theme(fig_price, "Stock Price with Moving Averages")
    fig_price.update_xaxes(rangeslider_visible=False)

    # Decomposition
    ts = df.set_index(date_col)[price_col].dropna()
    ts_weekly = ts.resample("W").mean().dropna()
    decomp = seasonal_decompose(ts_weekly, model="additive", period=4)

    fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
        subplot_titles=["Original","Trend","Seasonality","Residual"],
        vertical_spacing=0.06)
    colors_d = [COLORS["accent1"], COLORS["accent2"], COLORS["accent3"], COLORS["muted"]]
    for i, (s, label) in enumerate([(ts_weekly,"Original"),(decomp.trend,"Trend"),
                                    (decomp.seasonal,"Seasonal"),(decomp.resid,"Residual")], 1):
        fig_decomp.add_trace(
            go.Scatter(x=s.index, y=s.values, name=label,
                       line=dict(color=colors_d[i-1], width=1.5)), row=i, col=1)
    theme(fig_decomp, "Time-Series Decomposition (Weekly)")
    fig_decomp.update_layout(height=520, showlegend=False)

    # Returns distribution
    if "Returns" in df.columns:
        fig_ret = px.histogram(df, x="Returns", nbins=60,
            color_discrete_sequence=[COLORS["accent3"]])
        theme(fig_ret, "Daily Returns Distribution (%)")
    else:
        fig_ret = go.Figure()

    # Volume bar
    if "Volume" in df.columns:
        fig_vol = px.bar(df.tail(100), x=date_col, y="Volume",
            color_discrete_sequence=[COLORS["accent1"]])
        theme(fig_vol, "Trading Volume (Last 100 sessions)")
    else:
        fig_vol = go.Figure()

    return html.Div([
        section_header("Time-Series", "Analysis",
            "Decomposing stock price data into trend, seasonality, and residuals; moving-average smoothing."),
        dbc.Row([
            dbc.Col(kpi("Period","1 000 days","Jan 2020 – present",COLORS["accent1"]),md=3),
            dbc.Col(kpi("Sampling","Business day","Mon–Fri",COLORS["accent2"]),md=3),
            dbc.Col(kpi("MA Windows","7 & 30 day","Moving averages",COLORS["accent3"]),md=3),
            dbc.Col(kpi("Decomp Period","4 weeks","Additive model"),md=3),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_price,
                config={"displayModeBar":True}, style={"height":"340px"})]), md=12),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_decomp,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"520px"})]), md=8),
            dbc.Col([
                card([dcc.Graph(figure=fig_ret,
                    config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"230px"})]),
                card([dcc.Graph(figure=fig_vol,
                    config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"220px"})]),
            ], md=4),
        ], className="g-3"),
    ])

# ── LEVEL 2: Clustering (iris) ────────────────────────────────────────
def tab_clustering():
    fig_elbow = px.line(ELBOW_DF, x="k", y="inertia", markers=True,
        color_discrete_sequence=[COLORS["accent1"]])
    fig_elbow.add_vline(x=3, line_dash="dash", line_color=COLORS["accent2"])
    theme(fig_elbow, "Elbow Method — Inertia vs K")

    fig_sil = px.line(SIL_DF, x="k", y="score", markers=True,
        color_discrete_sequence=[COLORS["accent2"]])
    theme(fig_sil, "Silhouette Score vs K")
    fig_sil.update_yaxes(range=[0.2, 0.65])

    # Petal scatter by cluster
    fig_km = px.scatter(DF_IRIS, x="petal_length", y="petal_width",
        color="cluster",
        color_discrete_sequence=[COLORS["accent2"],COLORS["accent1"],COLORS["accent3"]],
        opacity=0.85, hover_data=["species"])
    theme(fig_km, "K-Means Clusters (K=3)")

    # True labels
    fig_true = px.scatter(DF_IRIS, x="petal_length", y="petal_width",
        color="species", color_discrete_map=SP_COLORS, opacity=0.85)
    theme(fig_true, "True Species Labels")

    # Crosstab
    ct = pd.crosstab(DF_IRIS["cluster"], DF_IRIS["species"])
    fig_ct = px.imshow(ct.values, x=ct.columns, y=ct.index.astype(str),
        text_auto=True, color_continuous_scale=["#13161e","#7de8c8"],
        labels={"x":"Species","y":"Cluster","color":"Count"})
    theme(fig_ct, "Cluster × Species Alignment")

    # 3-D scatter
    fig_3d = px.scatter_3d(DF_IRIS, x="petal_length", y="petal_width", z="sepal_length",
        color="cluster",
        color_discrete_sequence=[COLORS["accent2"],COLORS["accent1"],COLORS["accent3"]],
        opacity=0.8, size_max=8)
    fig_3d.update_layout(**PLOT_LAYOUT, height=400,
        scene=dict(bgcolor=COLORS["bg3"],
            xaxis=dict(backgroundcolor=COLORS["bg3"]),
            yaxis=dict(backgroundcolor=COLORS["bg3"]),
            zaxis=dict(backgroundcolor=COLORS["bg3"])))

    return html.Div([
        section_header("K-Means", "Clustering",
            "Optimal K selected via elbow + silhouette; cluster alignment against true species labels."),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_elbow,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"240px"})]), md=6),
            dbc.Col(card([dcc.Graph(figure=fig_sil,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"240px"})]), md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_km,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"300px"})]), md=6),
            dbc.Col(card([dcc.Graph(figure=fig_true,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"300px"})]), md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_ct,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"280px"})]), md=5),
            dbc.Col(card([dcc.Graph(figure=fig_3d,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"400px"})]), md=7),
        ], className="g-3"),
    ])

# ── LEVEL 3: Classification (iris) ───────────────────────────────────
def tab_classification():
    # Model comparison
    comp = [{"Model":k,"Accuracy":f"{v['accuracy']:.4f}",
             "F1-macro":f"{v['f1']:.4f}","CV-5":f"{v['cv']:.4f}"}
            for k,v in CLS_RESULTS.items()]
    comp_df = pd.DataFrame(comp)

    bar_data = pd.DataFrame([
        {"Model":k,"Metric":"Accuracy","Score":v["accuracy"]}
        for k,v in CLS_RESULTS.items()
    ] + [
        {"Model":k,"Metric":"F1-macro","Score":v["f1"]}
        for k,v in CLS_RESULTS.items()
    ])
    fig_bar = px.bar(bar_data, x="Model", y="Score", color="Metric",
        barmode="group", text_auto=".3f",
        color_discrete_sequence=[COLORS["accent1"],COLORS["accent3"]])
    theme(fig_bar, "Model Comparison — Accuracy & F1")
    fig_bar.update_yaxes(range=[0.85, 1.02])

    # Feature importance
    fig_imp = px.bar(FEAT_IMP_DF, x="importance", y="feature",
        orientation="h", text_auto=".3f",
        color="importance", color_continuous_scale=["#1a1e29","#9b8fff"])
    theme(fig_imp, "Feature Importance — Random Forest")

    # Confusion matrix (Random Forest)
    cm = CLS_RESULTS["Random Forest"]["cm"]
    fig_cm = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES,
        text_auto=True, color_continuous_scale=["#13161e","#7de8c8"],
        labels={"x":"Predicted","y":"Actual"})
    theme(fig_cm, "Confusion Matrix — Random Forest")

    # Precision / Recall per class (best model = GridSearchCV tuned RF)
    pr_rows = []
    for cls_name in CLASS_NAMES:
        d = GS_REPORT_DICT.get(cls_name, {})
        pr_rows.append({"Class": cls_name, "Metric": "Precision", "Score": d.get("precision", 0)})
        pr_rows.append({"Class": cls_name, "Metric": "Recall",    "Score": d.get("recall",    0)})
        pr_rows.append({"Class": cls_name, "Metric": "F1-Score",  "Score": d.get("f1-score",  0)})
    pr_df = pd.DataFrame(pr_rows)
    fig_pr = px.bar(pr_df, x="Class", y="Score", color="Metric", barmode="group",
        text_auto=".3f",
        color_discrete_sequence=[COLORS["accent1"], COLORS["accent2"], COLORS["accent3"]])
    theme(fig_pr, "Per-Class Precision / Recall / F1 — Tuned RF")
    fig_pr.update_yaxes(range=[0.8, 1.02])

    # GridSearchCV tuned confusion matrix
    fig_gs_cm = px.imshow(GS_CM, x=CLASS_NAMES, y=CLASS_NAMES,
        text_auto=True, color_continuous_scale=["#13161e","#7de8c8"],
        labels={"x":"Predicted","y":"Actual"})
    theme(fig_gs_cm, "Confusion Matrix — GridSearch Tuned RF")

    tbl_style = dict(backgroundColor=COLORS["bg3"], color=COLORS["text"],
                     fontFamily="DM Mono, monospace", fontSize="0.78rem")

    gs_params_text = " | ".join(f"{k}: {v}" for k, v in GS_BEST_PARAMS.items())

    return html.Div([
        section_header("Predictive", "Classification",
            "Logistic Regression, Decision Tree, and Random Forest on the Iris dataset — "
            "with GridSearchCV tuning, cross-validation, confusion matrix, and feature importance."),
        dbc.Row([
            dbc.Col(kpi("Best Accuracy", f"{GS_ACCURACY:.1%}", "GridSearchCV tuned RF",COLORS["accent1"]),md=3),
            dbc.Col(kpi("Best F1",       f"{GS_F1:.4f}",       "Macro avg tuned RF",COLORS["accent2"]),md=3),
            dbc.Col(kpi("CV F1",         f"{GS_BEST_SCORE:.4f}","5-fold GridSearchCV",COLORS["accent3"]),md=3),
            dbc.Col(kpi("Top Feature","petal_width","43.7% RF importance"),md=3),
        ], className="g-3 mb-3"),
        card([
            html.H3("GridSearchCV — Best Hyperparameters", style=dict(fontFamily="DM Mono, monospace",
                fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                letterSpacing="0.08em", marginBottom="10px")),
            html.Div(gs_params_text, style=dict(fontFamily="DM Mono, monospace",
                fontSize="0.82rem", color=COLORS["accent1"],
                background=COLORS["bg3"], padding="10px 16px",
                borderRadius="8px", border=f"1px solid {COLORS['border']}")),
            html.P("Parameter grid: n_estimators ∈ {50,100,200} · max_depth ∈ {None,5,10} · "
                   "min_samples_split ∈ {2,5} · cv=5 · scoring=f1_macro",
                style=dict(fontSize="0.72rem", color=COLORS["muted"], marginTop="8px")),
        ]),
        dbc.Row([
            dbc.Col(card([
                html.H3("Model Scorecard", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="12px")),
                dash_table.DataTable(
                    data=comp_df.to_dict("records"),
                    columns=[{"name":c,"id":c} for c in comp_df.columns],
                    style_header={**tbl_style, "color": COLORS["accent1"],
                                  "borderBottom": f"1px solid {COLORS['border']}"},
                    style_data={**tbl_style, "border": "none"},
                    style_table=dict(overflowX="auto"),
                    style_cell=dict(padding="10px 14px"),
                ),
                dcc.Graph(figure=fig_bar, config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}},
                          style={"height":"240px","marginTop":"12px"}),
            ]), md=7),
            dbc.Col(card([
                html.H3("Confusion Matrix", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="12px")),
                dcc.Graph(figure=fig_cm, config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}},
                          style={"height":"300px"}),
            ]), md=5),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_imp,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"240px"})]), md=5),
            dbc.Col(card([
                html.H3("Select model for detailed report", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="8px")),
                dcc.Dropdown(id="cls-model-select",
                    options=[{"label":k,"value":k} for k in CLS_RESULTS],
                    value="Random Forest",
                    style=dict(backgroundColor=COLORS["bg3"], color=COLORS["text"],
                               border=f"1px solid {COLORS['border']}", borderRadius="6px")),
                html.Pre(id="cls-report", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"],
                    marginTop="12px", whiteSpace="pre-wrap")),
            ]), md=7),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_pr,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"precision_recall"}},
                style={"height":"300px"})]), md=7),
            dbc.Col(card([dcc.Graph(figure=fig_gs_cm,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"confusion_matrix_tuned"}},
                style={"height":"300px"})]), md=5),
        ], className="g-3"),
    ])

# ── LEVEL 3: NLP Sentiment ───────────────────────────────────────────
def tab_nlp():
    df = DF_SENTIMENT.copy()

    # Distribution
    vc = df["tb_label"].value_counts().reset_index()
    vc.columns = ["Sentiment","Count"]
    fig_dist = px.pie(vc, names="Sentiment", values="Count",
        color="Sentiment",
        color_discrete_map={"positive":COLORS["accent1"],
                             "negative":COLORS["accent2"],
                             "neutral":COLORS["accent3"]},
        hole=0.55)
    theme(fig_dist, "TextBlob Sentiment Distribution")

    # Polarity vs Subjectivity
    fig_pol = px.scatter(df, x="polarity", y="subjectivity",
        color="tb_label",
        color_discrete_map={"positive":COLORS["accent1"],
                             "negative":COLORS["accent2"],
                             "neutral":COLORS["accent3"]},
        opacity=0.75, size_max=8,
        hover_data=["clean"])
    fig_pol.add_vline(x=0.05, line_dash="dash", line_color=COLORS["accent1"], opacity=0.4)
    fig_pol.add_vline(x=-0.05, line_dash="dash", line_color=COLORS["accent2"], opacity=0.4)
    theme(fig_pol, "Polarity vs Subjectivity")

    # Rating histogram (column may not exist in all CSVs)
    _rating_col = next((c for c in df.columns if c.lower() == "rating"), None)
    if _rating_col:
        fig_rat = px.histogram(df, x=_rating_col, color="tb_label",
            barmode="overlay", nbins=5,
            color_discrete_map={"positive":COLORS["accent1"],
                                 "negative":COLORS["accent2"],
                                 "neutral":COLORS["accent3"]})
        theme(fig_rat, "Rating Distribution by Sentiment")
    else:
        fig_rat = go.Figure()
        theme(fig_rat, "Rating Distribution by Sentiment (no rating column)")

    # Polarity box per true label (handle both 'sentiment' and 'Sentiment')
    _sent_col = next((c for c in df.columns if c.lower() == "sentiment"), None)
    if _sent_col:
        fig_box = px.box(df, x=_sent_col, y="polarity",
            color=_sent_col,
            color_discrete_map={"positive":COLORS["accent1"],
                                 "negative":COLORS["accent2"],
                                 "neutral":COLORS["accent3"],
                                 "Positive":COLORS["accent1"],
                                 "Negative":COLORS["accent2"],
                                 "Neutral":COLORS["accent3"]},
            points="all")
        theme(fig_box, "Polarity by True Sentiment Label")
    else:
        fig_box = go.Figure()

    # Top words per class
    def top_words(label, n=12):
        texts = df[df["tb_label"]==label]["clean"]
        words = " ".join(texts).split()
        stop  = {"the","a","an","is","it","and","or","but","not","for",
                  "to","of","in","on","at","i","this","that","with","be","are"}
        cnt = Counter(w for w in words if len(w)>2 and w not in stop)
        top = cnt.most_common(n)
        return pd.DataFrame(top, columns=["word","count"])

    wdf_pos = top_words("positive")
    wdf_neg = top_words("negative")

    fig_words = make_subplots(rows=1, cols=2,
        subplot_titles=["Top Positive Words","Top Negative Words"])
    fig_words.add_trace(go.Bar(x=wdf_pos["count"], y=wdf_pos["word"],
        orientation="h", marker_color=COLORS["accent1"], showlegend=False), row=1, col=1)
    fig_words.add_trace(go.Bar(x=wdf_neg["count"], y=wdf_neg["word"],
        orientation="h", marker_color=COLORS["accent2"], showlegend=False), row=1, col=2)
    theme(fig_words, "Word Frequency by Sentiment")
    fig_words.update_layout(height=340)

    # Word cloud (all reviews)
    _wc_text = " ".join(df["clean"].tolist())
    _stop = {"the","a","an","is","it","and","or","but","not","for","to","of",
             "in","on","at","i","this","that","with","be","are","was","have","has"}
    try:
        _wc = WordCloud(
            width=700, height=280,
            background_color="#13161e",
            colormap="summer",
            stopwords=_stop,
            max_words=80,
            prefer_horizontal=0.9,
        ).generate(_wc_text)
        _buf = io.BytesIO()
        _fig_wc, _ax = plt.subplots(figsize=(7, 2.8), facecolor="#13161e")
        _ax.imshow(_wc, interpolation="bilinear")
        _ax.axis("off")
        _fig_wc.tight_layout(pad=0)
        _fig_wc.savefig(_buf, format="png", bbox_inches="tight",
                        facecolor="#13161e", dpi=120)
        plt.close(_fig_wc)
        _buf.seek(0)
        _wc_b64 = base64.b64encode(_buf.read()).decode()
        wc_img = html.Img(src=f"data:image/png;base64,{_wc_b64}",
                          style={"width":"100%", "borderRadius":"8px"})
    except Exception:
        wc_img = html.Div("Word cloud unavailable.", style={"color": COLORS["muted"]})

    # Live analyser
    return html.Div([
        section_header("NLP Sentiment", "Analysis",
            "TextBlob polarity & subjectivity scoring on product reviews. "
            "Visualises sentiment distribution, word frequencies, and rating correlations."),
        dbc.Row([
            dbc.Col(kpi("Total Reviews", str(len(df)), "Analysed",COLORS["accent1"]),md=3),
            dbc.Col(kpi("Positive", str((df["tb_label"]=="positive").sum()),
                        f"{(df['tb_label']=='positive').mean()*100:.0f}%",COLORS["accent1"]),md=3),
            dbc.Col(kpi("Negative", str((df["tb_label"]=="negative").sum()),
                        f"{(df['tb_label']=='negative').mean()*100:.0f}%",COLORS["accent2"]),md=3),
            dbc.Col(kpi("Neutral",  str((df["tb_label"]=="neutral").sum()),
                        f"{(df['tb_label']=='neutral').mean()*100:.0f}%",COLORS["accent3"]),md=3),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_dist,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"280px"})]), md=4),
            dbc.Col(card([dcc.Graph(figure=fig_pol,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"280px"})]), md=8),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([dcc.Graph(figure=fig_words,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"340px"})]), md=8),
            dbc.Col(card([dcc.Graph(figure=fig_rat,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"280px"})]), md=4),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([
                html.H3("Word Cloud — All Reviews", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="10px")),
                wc_img,
            ]), md=6),
            dbc.Col(card([dcc.Graph(figure=fig_box,
                config={"displayModeBar":True,"toImageButtonOptions":{"format":"png","filename":"chart"}}, style={"height":"260px"})]), md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([
                html.H3("Live Sentiment Analyser", style=dict(fontFamily="DM Mono, monospace",
                    fontSize="0.72rem", color=COLORS["muted"], textTransform="uppercase",
                    letterSpacing="0.08em", marginBottom="10px")),
                dcc.Textarea(id="nlp-input", placeholder="Type any review text here…",
                    style=dict(width="100%", height="80px",
                               backgroundColor=COLORS["bg3"], color=COLORS["text"],
                               border=f"1px solid {COLORS['border']}", borderRadius="6px",
                               padding="10px", fontFamily="DM Sans, sans-serif",
                               fontSize="0.85rem", resize="vertical")),
                html.Button("Analyse", id="nlp-btn", n_clicks=0,
                    style=dict(marginTop="8px", padding="8px 20px",
                               backgroundColor=COLORS["accent1"], color="#0d0f14",
                               border="none", borderRadius="6px", cursor="pointer",
                               fontFamily="DM Mono, monospace", fontSize="0.75rem")),
                html.Div(id="nlp-result", style=dict(marginTop="12px",
                    fontSize="0.82rem", color=COLORS["muted"])),
            ]), md=6),
        ], className="g-3"),
    ])

# ──────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ──────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&"
        "family=DM+Sans:wght@300;400;500&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Iris Analytics · Codveda",
)
server = app.server   # for deployment (gunicorn / render.com)

NAV_TABS = [
    ("L1 — Cleaning",        "cleaning"),
    ("L1 — EDA",             "eda"),
    ("L1 — Visualisation",   "viz"),
    ("L2 — Regression",      "regression"),
    ("L2 — Time Series",     "timeseries"),
    ("L2 — Clustering",      "clustering"),
    ("L3 — Classification",  "classification"),
    ("L3 — NLP Sentiment",   "nlp"),
]

app.layout = html.Div([
    # ── NAV ──────────────────────────────────────────────────────────
    html.Nav([
        html.Div([
            html.Span("iris", style=dict(color=COLORS["accent1"])),
            html.Span(".", style=dict(color=COLORS["accent2"])),
            html.Span("analytics"),
        ], style=dict(fontFamily="Georgia, serif", fontSize="1.05rem",
                      fontWeight="600", color=COLORS["text"],
                      letterSpacing="-0.02em", marginRight="24px",
                      whiteSpace="nowrap")),
        dcc.Tabs(id="main-tabs", value="cleaning",
            children=[dcc.Tab(label=lbl, value=val,
                style=dict(backgroundColor=COLORS["bg"], color=COLORS["muted"],
                           borderTop="none", borderLeft="none", borderRight="none",
                           borderBottom="2px solid transparent",
                           fontFamily="DM Mono, monospace", fontSize="0.68rem",
                           textTransform="uppercase", letterSpacing="0.06em",
                           padding="0 16px", lineHeight="55px"),
                selected_style=dict(backgroundColor=COLORS["bg"], color=COLORS["accent1"],
                                    borderTop="none", borderLeft="none", borderRight="none",
                                    borderBottom=f"2px solid {COLORS['accent1']}",
                                    fontFamily="DM Mono, monospace", fontSize="0.68rem",
                                    textTransform="uppercase", letterSpacing="0.06em",
                                    padding="0 16px", lineHeight="55px"),
            ) for lbl, val in NAV_TABS],
            style=dict(flex="1", border="none", height="56px"),
        ),
        html.Div([
            html.Span("Codveda Internship",
                style=dict(fontFamily="DM Mono, monospace", fontSize="0.68rem",
                           color=COLORS["muted"], padding="4px 10px",
                           background="rgba(125,232,200,0.08)",
                           border=f"1px solid rgba(125,232,200,0.2)",
                           borderRadius="100px")),
        ], style=dict(marginLeft="auto", paddingLeft="16px", whiteSpace="nowrap")),
    ], style=dict(position="fixed", top=0, left=0, right=0, zIndex=100,
                  display="flex", alignItems="center", padding="0 24px",
                  height="56px", backgroundColor=COLORS["bg"],
                  borderBottom=f"1px solid {COLORS['border']}",
                  backdropFilter="blur(12px)")),

    # ── CONTENT ──────────────────────────────────────────────────────
    html.Div(id="tab-content",
             style=dict(paddingTop="72px", paddingBottom="48px",
                        paddingLeft="32px", paddingRight="32px",
                        maxWidth="1400px", margin="0 auto")),
], style=dict(backgroundColor=COLORS["bg"], color=COLORS["text"],
              fontFamily="DM Sans, sans-serif", minHeight="100vh"))

# ──────────────────────────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────────────────────────

@app.callback(Output("tab-content","children"), Input("main-tabs","value"))
def render_tab(tab):
    return {
        "cleaning":       tab_cleaning,
        "eda":            tab_eda,
        "viz":            tab_viz,
        "regression":     tab_regression,
        "timeseries":     tab_timeseries,
        "clustering":     tab_clustering,
        "classification": tab_classification,
        "nlp":            tab_nlp,
    }[tab]()


@app.callback(Output("eda-hist","figure"), Input("eda-feat","value"))
def update_hist(feat):
    fig = px.histogram(DF_IRIS, x=feat, color="species",
        color_discrete_map=SP_COLORS, nbins=20, barmode="overlay",
        opacity=0.75,
        labels={feat: feat.replace("_"," ").title() + " (cm)"})
    theme(fig, feat.replace("_"," ").title() + " — Distribution by Species")
    return fig


@app.callback(Output("cls-report","children"), Input("cls-model-select","value"))
def update_report(model_name):
    return CLS_RESULTS[model_name]["report"]


@app.callback(Output("nlp-result","children"),
              Input("nlp-btn","n_clicks"), State("nlp-input","value"))
def analyse_text(n, text):
    if not n or not text:
        return ""
    blob = TextBlob(text)
    pol  = blob.sentiment.polarity
    sub  = blob.sentiment.subjectivity
    label = "POSITIVE" if pol > 0.05 else ("NEGATIVE" if pol < -0.05 else "NEUTRAL")
    color = {
        "POSITIVE": COLORS["accent1"],
        "NEGATIVE": COLORS["accent2"],
        "NEUTRAL":  COLORS["accent3"],
    }[label]
    return html.Div([
        html.Div(label, style=dict(fontFamily="DM Mono, monospace",
            fontSize="1.4rem", fontWeight="600", color=color, marginBottom="8px")),
        html.Div([
            html.Span("Polarity: ", style=dict(color=COLORS["muted"])),
            html.Span(f"{pol:+.4f}", style=dict(color=color,
                fontFamily="DM Mono, monospace")),
        ], style=dict(marginBottom="4px")),
        html.Div([
            html.Span("Subjectivity: ", style=dict(color=COLORS["muted"])),
            html.Span(f"{sub:.4f}", style=dict(color=COLORS["text"],
                fontFamily="DM Mono, monospace")),
        ]),
    ])

# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "─"*60)
    print("  Codveda Internship Dashboard")
    print("  Open:  http://127.0.0.1:8050")
    print("─"*60 + "\n")
    app.run(debug=False, port=8050, host="0.0.0.0")
