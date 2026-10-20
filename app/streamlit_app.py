"""
DataSentinel Dashboard Avanzado (Streamlit).
Permite explorar la señal, inyectar perturbaciones y morfologías sintéticas,
simular costes de negocio y visualizar interactivamente las Curvas Precision-Recall (PR-AUC).
"""
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.detectors.baseline_statistical import RollingZScoreDetector, RollingIQRDetector
from src.detectors.isolation_forest_detector import TimeSeriesIsolationForest
from src.detectors.lof_detector import TimeSeriesLOF
from src.detectors.arima_detector import AutoRegressiveAnomalyDetector
from src.evaluation.metrics import evaluate_pointwise, evaluate_event_aware
from src.evaluation.cost_matrix import compute_business_costs
from src.perturbations.generator import inject_gaussian_noise, inject_concept_drift
from src.perturbations.synthetic_anomalies import inject_point_spikes, inject_sensor_freeze, inject_pattern_inversion
from src.data.download_ucr_sample import parse_ucr_filename, SAMPLE_FILENAME

st.set_page_config(page_title="DataSentinel Research Suite", layout="wide")

st.title("🛡️ DataSentinel: Suite de Investigación y Benchmark de Anomalías")
st.markdown("Plataforma interactiva para evaluar la estabilidad de detectores de series temporales bajo perturbaciones y morfologías controladas.")

DATA_FILE = os.path.join(ROOT_DIR, "data", "raw", SAMPLE_FILENAME)

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None, None
    meta = parse_ucr_filename(SAMPLE_FILENAME)
    df = pd.read_csv(DATA_FILE, header=None, names=["value"])
    return df, meta

df, meta = load_data()

if df is None:
    st.error(f"No se encontró {DATA_FILE}. Ejecuta primero: python -m src.data.download_ucr_sample")
    st.stop()

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

# Preparación de datos
train_split = meta["train_end"]
train_series = df["value"].iloc[:train_split].reset_index(drop=True)
base_test = df["value"].iloc[train_split:].reset_index(drop=True)

# Selección de señal según el modo
if anomaly_mode == "Anomalía Real UCR (Ground Truth)":
    y_true_full = np.zeros(len(df), dtype=int)
    y_true_full[meta["anomaly_start"]:meta["anomaly_end"]] = 1
    test_y_true = y_true_full[train_split:]
    current_test = base_test.copy()
elif anomaly_mode == "Picos Aislados Sintéticos (Spikes)":
    current_test, test_y_true = inject_point_spikes(base_test, n_spikes=30, magnitude=4.0)
elif anomaly_mode == "Congelación de Sensor (Flatline)":
    current_test, test_y_true = inject_sensor_freeze(base_test, start_idx=3000, duration=400)
else:
    current_test, test_y_true = inject_pattern_inversion(base_test, start_idx=6000, duration=500)

# Perturbaciones
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
        det = AutoRegressiveAnomalyDetector(lags=10, threshold_sigma=3.5)
        det.fit(train_series)
        pred_df = det.predict(current_test)
    elif selected_model == "Isolation Forest":
        det = TimeSeriesIsolationForest(window_size=50, contamination=0.02, random_state=42)
        det.fit(train_series)
        pred_df = det.predict(current_test)
    else:
        det = TimeSeriesLOF(window_size=50, n_neighbors=30, contamination=0.02)
        det.fit(train_series)
        pred_df = det.predict(current_test)

    pw = evaluate_pointwise(test_y_true, pred_df["anomaly_pred"].values, pred_df["score"].values)
    ev = evaluate_event_aware(test_y_true, pred_df["anomaly_pred"].values)
    costs = compute_business_costs(test_y_true, pred_df["anomaly_pred"].values, cost_fp=cost_fp, cost_fn=cost_fn)

# Métricas
col_a, col_b = st.columns([3, 2])
with col_a:
    st.subheader("Métricas de Rendimiento Técnico")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1-Score", f"{pw['f1']:.3f}")
    m2.metric("Precision", f"{pw['precision']:.3f}")
    m3.metric("Recall", f"{pw['recall']:.3f}")
    m4.metric("PR-AUC", f"{pw['pr_auc']:.3f}")

with col_b:
    st.subheader("Impacto Económico")
    e1, e2 = st.columns(2)
    e1.metric("Falsas Alarmas", pw["total_false_alarms"])
    e2.metric("Coste Total (TEC)", f"{costs['total_expected_cost_eur']:,.0f} €")

# Visualización temporal
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

alerts = np.where(pred_df["anomaly_pred"] == 1)[0]
if len(alerts) > 0:
    fig.add_trace(go.Scatter(
        x=alerts, y=current_test.iloc[alerts].values,
        mode='markers', name=f'Alertas ({len(alerts)})',
        marker=dict(color='red', size=4)
    ))

fig.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Tiempo", yaxis_title="Valor")
st.plotly_chart(fig, use_container_width=True)

# Curva Precision-Recall
st.subheader("Curva Precision-Recall (Evaluación Continua)")
precisions, recalls, _ = precision_recall_curve(test_y_true, pred_df["score"].values)
pr_auc_score = average_precision_score(test_y_true, pred_df["score"].values)

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=recalls, y=precisions, mode='lines', name=f'PR Curve (AUC = {pr_auc_score:.3f})', line=dict(color='#d9534f', width=2)))
fig_pr.update_layout(
    height=350,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="Recall (Sensibilidad)",
    yaxis_title="Precision",
    xaxis=dict(range=[0, 1.05]),
    yaxis=dict(range=[0, 1.05])
)
st.plotly_chart(fig_pr, use_container_width=True)
