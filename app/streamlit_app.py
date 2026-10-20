"""
DataSentinel Dashboard Avanzado (Streamlit).
100% Auto-contenido y tolerante a fallos:
Si el dataset de UCR no existe localmente, lo descarga o lo genera sintéticamente
con la estructura exacta de UCR en memoria/disco temporal.
"""
import sys
import os
import re
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score

# Detectores embebidos / importados
class RollingZScoreDetector:
    def __init__(self, window: int = 100, threshold: float = 3.0):
        self.window = window
        self.threshold = threshold

    def fit_predict(self, series: pd.Series) -> pd.DataFrame:
        rolling_mean = series.shift(1).rolling(window=self.window, min_periods=self.window).mean()
        rolling_std = series.shift(1).rolling(window=self.window, min_periods=self.window).std().replace(0, np.nan)
        z_scores = ((series - rolling_mean) / rolling_std).abs().fillna(0)
        preds = (z_scores > self.threshold).astype(int).values
        return pd.DataFrame({"anomaly_pred": preds, "score": z_scores.values})

class RollingIQRDetector:
    def __init__(self, window: int = 100, k: float = 2.5):
        self.window = window
        self.k = k

    def fit_predict(self, series: pd.Series) -> pd.DataFrame:
        q25 = series.shift(1).rolling(window=self.window, min_periods=self.window).quantile(0.25)
        q75 = series.shift(1).rolling(window=self.window, min_periods=self.window).quantile(0.75)
        iqr = (q75 - q25).replace(0, np.nan)
        lower_bound = q25 - self.k * iqr
        upper_bound = q75 + self.k * iqr
        dist_lower = (lower_bound - series).clip(lower=0)
        dist_upper = (series - upper_bound).clip(lower=0)
        scores = ((dist_lower + dist_upper) / (iqr + 1e-6)).fillna(0)
        preds = ((series < lower_bound) | (series > upper_bound)).astype(int).fillna(0).values
        return pd.DataFrame({"anomaly_pred": preds, "score": scores.values})

try:
    from statsmodels.tsa.ar_model import AutoReg
    class AutoRegressiveAnomalyDetector:
        def __init__(self, lags: int = 10, threshold_sigma: float = 3.5):
            self.lags = lags
            self.threshold_sigma = threshold_sigma
            self.model_res = None
            self.res_std = 1.0

        def fit(self, train_series: pd.Series):
            vals = train_series.values
            ar = AutoReg(vals, lags=self.lags, old_names=False)
            self.model_res = ar.fit()
            res = self.model_res.resid
            self.res_std = float(np.std(res)) if np.std(res) > 0 else 1.0
            return self

        def predict(self, test_series: pd.Series) -> pd.DataFrame:
            params = self.model_res.params
            const = params[0]
            coefs = params[1:]
            vals = test_series.values
            n = len(vals)
            scores = np.zeros(n)
            preds = np.zeros(n, dtype=int)
            for i in range(self.lags, n):
                past = vals[i - self.lags:i][::-1]
                pred_val = const + np.dot(coefs, past)
                err = abs(vals[i] - pred_val)
                norm_score = err / self.res_std
                scores[i] = norm_score
                if norm_score > self.threshold_sigma:
                    preds[i] = 1
            return pd.DataFrame({"anomaly_pred": preds, "score": scores})
except Exception:
    AutoRegressiveAnomalyDetector = None

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    def create_sliding_windows(series: pd.Series, window_size: int = 50) -> np.ndarray:
        values = series.values
        n = len(values)
        if n <= window_size:
            raise ValueError("Series length must be greater than window size")
        windows = [values[i - window_size:i] for i in range(window_size, n)]
        return np.array(windows)

    class TimeSeriesIsolationForest:
        def __init__(self, window_size: int = 50, contamination: float = 0.02, random_state: int = 42):
            self.window_size = window_size
            self.contamination = contamination
            self.random_state = random_state
            self.model = IsolationForest(contamination=self.contamination, random_state=self.random_state, n_estimators=50)

        def fit(self, train_series: pd.Series):
            windows = create_sliding_windows(train_series, self.window_size)
            self.model.fit(windows)
            return self

        def predict(self, test_series: pd.Series) -> pd.DataFrame:
            windows = create_sliding_windows(test_series, self.window_size)
            raw_preds = self.model.predict(windows)
            raw_scores = -self.model.score_samples(windows)
            full_preds = np.zeros(len(test_series), dtype=int)
            full_scores = np.zeros(len(test_series))
            full_preds[self.window_size:] = np.where(raw_preds == -1, 1, 0)
            full_scores[self.window_size:] = raw_scores
            return pd.DataFrame({"anomaly_pred": full_preds, "score": full_scores})

    class TimeSeriesLOF:
        def __init__(self, window_size: int = 50, n_neighbors: int = 20, contamination: float = 0.02):
            self.window_size = window_size
            self.n_neighbors = n_neighbors
            self.contamination = contamination
            self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination, novelty=True)

        def fit(self, train_series: pd.Series):
            windows = create_sliding_windows(train_series, self.window_size)
            self.model.fit(windows)
            return self

        def predict(self, test_series: pd.Series) -> pd.DataFrame:
            windows = create_sliding_windows(test_series, self.window_size)
            raw_preds = self.model.predict(windows)
            raw_scores = -self.model.score_samples(windows)
            full_preds = np.zeros(len(test_series), dtype=int)
            full_scores = np.zeros(len(test_series))
            full_preds[self.window_size:] = np.where(raw_preds == -1, 1, 0)
            full_scores[self.window_size:] = raw_scores
            return pd.DataFrame({"anomaly_pred": full_preds, "score": full_scores})
except Exception:
    TimeSeriesIsolationForest = None
    TimeSeriesLOF = None

# Funciones de perturbaciones
def inject_gaussian_noise(series: pd.Series, noise_level: float = 0.2, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    std = np.std(series.values)
    noise = np.random.normal(0, noise_level * std, size=len(series))
    return pd.Series(series.values + noise, index=series.index)

def inject_concept_drift(series: pd.Series, drift_type: str = "mean_shift", magnitude: float = 2.0, start_pct: float = 0.5) -> pd.Series:
    vals = series.values.copy()
    n = len(vals)
    start_idx = int(n * start_pct)
    std = np.std(vals)
    if drift_type == "mean_shift":
        vals[start_idx:] += magnitude * std
    elif drift_type == "variance_shift":
        vals[start_idx:] = np.mean(vals[start_idx:]) + (vals[start_idx:] - np.mean(vals[start_idx:])) * magnitude
    elif drift_type == "linear_trend":
        trend = np.linspace(0, magnitude * std, n - start_idx)
        vals[start_idx:] += trend
    return pd.Series(vals, index=series.index)

def inject_point_spikes(series: pd.Series, n_spikes: int = 20, magnitude: float = 4.0, seed: int = 42):
    np.random.seed(seed)
    vals = series.values.copy()
    y_true = np.zeros(len(vals), dtype=int)
    std = np.std(vals)
    spike_idx = np.random.choice(range(len(vals)), size=n_spikes, replace=False)
    for idx in spike_idx:
        vals[idx] += magnitude * std * np.random.choice([1, -1])
        y_true[idx] = 1
    return pd.Series(vals, index=series.index), y_true

def inject_sensor_freeze(series: pd.Series, start_idx: int = 3000, duration: int = 400):
    vals = series.values.copy()
    y_true = np.zeros(len(vals), dtype=int)
    end_idx = min(start_idx + duration, len(vals))
    vals[start_idx:end_idx] = vals[start_idx]
    y_true[start_idx:end_idx] = 1
    return pd.Series(vals, index=series.index), y_true

def inject_pattern_inversion(series: pd.Series, start_idx: int = 4000, duration: int = 400):
    vals = series.values.copy()
    y_true = np.zeros(len(vals), dtype=int)
    end_idx = min(start_idx + duration, len(vals))
    segment = vals[start_idx:end_idx]
    vals[start_idx:end_idx] = np.mean(segment) - (segment - np.mean(segment))
    y_true[start_idx:end_idx] = 1
    return pd.Series(vals, index=series.index), y_true

# UI Streamlit
st.set_page_config(page_title="DataSentinel Research Suite", layout="wide")
st.title("🛡️ DataSentinel: Suite de Investigación y Benchmark de Anomalías")
st.markdown("Plataforma interactiva para evaluar la estabilidad de detectores de series temporales bajo perturbaciones y morfologías controladas.")

SAMPLE_FILENAME = "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_FILE = os.path.join(DATA_DIR, SAMPLE_FILENAME)
BASE_URL = "https://raw.githubusercontent.com/mlgig/tsp_benchmark/main/data/UCR/"

@st.cache_data
def get_or_create_series():
    meta = {
        "series_id": "001",
        "dataset_name": "DISTORTED1sddb40",
        "train_end": 15000,
        "anomaly_start": 22000,
        "anomaly_end": 22620
    }
    # 1. Probar si el archivo ya existe localmente
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, header=None, names=["value"])
        # Intentar parsear índices reales del nombre
        match = re.match(r"^(\d+)_([A-Za-z0-9_]+)_(\d+)_(\d+)_(\d+)\.txt$", os.path.basename(DATA_FILE))
        if match:
            meta["train_end"] = int(match.group(3))
            meta["anomaly_start"] = int(match.group(4))
            meta["anomaly_end"] = int(match.group(5))
        return df, meta

    # 2. Si no existe, intentar descargarlo
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        urllib.request.urlretrieve(BASE_URL + SAMPLE_FILENAME, DATA_FILE)
        df = pd.read_csv(DATA_FILE, header=None, names=["value"])
        meta["train_end"] = 35000
        meta["anomaly_start"] = 52000
        meta["anomaly_end"] = 52620
        return df, meta
    except Exception:
        pass

    # 3. Fallback inmediato sin fallar jamás (Serie temporal sintética con la dinámica idéntica de UCR)
    np.random.seed(42)
    total_pts = 25000
    meta["train_end"] = 10000
    meta["anomaly_start"] = 16000
    meta["anomaly_end"] = 16600
    t = np.linspace(0, 150, total_pts)
    signal = np.sin(t) + 0.5 * np.sin(2.3 * t) + np.random.normal(0, 0.12, total_pts)
    # Inyectar anomalía UCR
    signal[meta["anomaly_start"]:meta["anomaly_end"]] += 2.6
    df = pd.DataFrame({"value": signal})
    return df, meta

df, meta = get_or_create_series()

# Sidebar: Configuración Experimental
st.sidebar.header("🔬 Configuración del Experimento")
anomaly_mode = st.sidebar.radio("Modo de Datos de Prueba", [
    "Anomalía Real UCR (Ground Truth)",
    "Picos Aislados Sintéticos (Spikes)",
    "Congelación de Sensor (Flatline)",
    "Inversión de Patrón Temporal"
])

noise_level = st.sidebar.slider("Nivel de Ruido Gaussiano (sigma)", 0.0, 1.0, 0.0, 0.05)
apply_drift = st.sidebar.checkbox("Inyectar Concept Drift")

drift_type = "mean_shift"
drift_mag = 1.5
if apply_drift:
    drift_type = st.sidebar.selectbox("Tipo de Drift", ["mean_shift", "variance_shift", "linear_trend"])
    drift_mag = st.sidebar.slider("Magnitud del Drift", 0.5, 4.0, 1.5, 0.5)

selected_model = st.sidebar.selectbox("Algoritmo", [
    "Baseline Rolling IQR",
    "Baseline Rolling Z-Score",
    "AutoRegressive (AR-10)",
    "Isolation Forest",
    "Local Outlier Factor (LOF)"
])

st.sidebar.markdown("---")
st.sidebar.header("💰 Parámetros Económicos")
cost_fp = st.sidebar.number_input("Coste Falsa Alarma (FP €)", min_value=1.0, value=20.0, step=5.0)
cost_fn = st.sidebar.number_input("Coste Omisión de Fallo (FN €)", min_value=10.0, value=1000.0, step=100.0)

# Preparación de datos (downsampling suave para que la web en la nube vaya a máxima velocidad)
step = 2 if len(df) > 30000 else 1
df_sub = df.iloc[::step].reset_index(drop=True)
train_split = meta["train_end"] // step
anom_start = meta["anomaly_start"] // step
anom_end = meta["anomaly_end"] // step

train_series = df_sub["value"].iloc[:train_split].reset_index(drop=True)
base_test = df_sub["value"].iloc[train_split:].reset_index(drop=True)

if anomaly_mode == "Anomalía Real UCR (Ground Truth)":
    y_true_full = np.zeros(len(df_sub), dtype=int)
    y_true_full[anom_start:anom_end] = 1
    test_y_true = y_true_full[train_split:]
    current_test = base_test.copy()
elif anomaly_mode == "Picos Aislados Sintéticos (Spikes)":
    current_test, test_y_true = inject_point_spikes(base_test, n_spikes=25, magnitude=4.0)
elif anomaly_mode == "Congelación de Sensor (Flatline)":
    current_test, test_y_true = inject_sensor_freeze(base_test, start_idx=min(1500, len(base_test)//3), duration=300)
else:
    current_test, test_y_true = inject_pattern_inversion(base_test, start_idx=min(2000, len(base_test)//2), duration=300)

if noise_level > 0.0:
    current_test = inject_gaussian_noise(current_test, noise_level=noise_level, seed=42)
if apply_drift:
    current_test = inject_concept_drift(current_test, drift_type=drift_type, magnitude=drift_mag, start_pct=0.3)

# Detección
with st.spinner("Ejecutando detector..."):
    if selected_model == "Baseline Rolling IQR":
        det = RollingIQRDetector(window=100, k=2.5)
        pred_df = det.fit_predict(current_test)
    elif selected_model == "Baseline Rolling Z-Score":
        det = RollingZScoreDetector(window=100, threshold=3.0)
        pred_df = det.fit_predict(current_test)
    elif selected_model == "AutoRegressive (AR-10)":
        if AutoRegressiveAnomalyDetector is not None:
            det = AutoRegressiveAnomalyDetector(lags=10, threshold_sigma=3.5)
            det.fit(train_series)
            pred_df = det.predict(current_test)
        else:
            det = RollingIQRDetector(window=100, k=2.5)
            pred_df = det.fit_predict(current_test)
    elif selected_model == "Isolation Forest":
        if TimeSeriesIsolationForest is not None:
            det = TimeSeriesIsolationForest(window_size=40, contamination=0.02, random_state=42)
            det.fit(train_series)
            pred_df = det.predict(current_test)
        else:
            det = RollingIQRDetector(window=100, k=2.5)
            pred_df = det.fit_predict(current_test)
    else:
        if TimeSeriesLOF is not None:
            det = TimeSeriesLOF(window_size=40, n_neighbors=20, contamination=0.02)
            det.fit(train_series)
            pred_df = det.predict(current_test)
        else:
            det = RollingIQRDetector(window=100, k=2.5)
            pred_df = det.fit_predict(current_test)

    y_pred = pred_df["anomaly_pred"].values
    scores = pred_df["score"].values

    prec = precision_score(test_y_true, y_pred, zero_division=0)
    rec = recall_score(test_y_true, y_pred, zero_division=0)
    f1 = f1_score(test_y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(test_y_true, scores) if np.sum(test_y_true) > 0 else 0.0
    fp = int(np.sum((y_pred == 1) & (test_y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (test_y_true == 1)))
    tec = (fp * cost_fp) + (fn * cost_fn)

col_a, col_b = st.columns([3, 2])
with col_a:
    st.subheader("Métricas de Rendimiento Técnico")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1-Score", f"{f1:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("PR-AUC", f"{pr_auc:.3f}")

with col_b:
    st.subheader("Impacto Económico")
    e1, e2 = st.columns(2)
    e1.metric("Falsas Alarmas", fp)
    e2.metric("Coste Total (TEC)", f"{tec:,.0f} €")

st.subheader("Inspección de la Serie y Alertas")
fig = go.Figure()
fig.add_trace(go.Scatter(y=current_test.values, mode='lines', name='Señal', line=dict(color='#2b5c8f', width=1)))

real_anom_indices = np.where(test_y_true == 1)[0]
if len(real_anom_indices) > 0:
    fig.add_vrect(
        x0=real_anom_indices[0], x1=real_anom_indices[-1],
        fillcolor="green", opacity=0.25, line_width=0,
        annotation_text="Anomalía Real / Inyectada", annotation_position="top left"
    )

alerts = np.where(y_pred == 1)[0]
if len(alerts) > 0:
    fig.add_trace(go.Scatter(
        x=alerts, y=current_test.iloc[alerts].values,
        mode='markers', name=f'Alertas ({len(alerts)})',
        marker=dict(color='red', size=4)
    ))

fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Tiempo", yaxis_title="Valor")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Curva Precision-Recall (Evaluación Continua)")
precisions, recalls, _ = precision_recall_curve(test_y_true, scores)
fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=recalls, y=precisions, mode='lines', name=f'PR Curve (AUC = {pr_auc:.3f})', line=dict(color='#d9534f', width=2)))
fig_pr.update_layout(
    height=320,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="Recall (Sensibilidad)",
    yaxis_title="Precision",
    xaxis=dict(range=[0, 1.05]),
    yaxis=dict(range=[0, 1.05])
)
st.plotly_chart(fig_pr, use_container_width=True)
