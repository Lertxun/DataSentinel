"""
Experimento de Sensibilidad por Morfología de Anomalía:
Picos Aislados (Spikes) vs Congelación de Señal (Sensor Freeze) vs Inversión de Patrón.
Demuestra qué familias de algoritmos fallan ante anomalías que no alteran la amplitud.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.detectors.baseline_statistical import RollingZScoreDetector, RollingIQRDetector
from src.detectors.isolation_forest_detector import TimeSeriesIsolationForest
from src.detectors.lof_detector import TimeSeriesLOF
from src.detectors.arima_detector import AutoRegressiveAnomalyDetector
from src.evaluation.metrics import evaluate_pointwise, evaluate_event_aware
from src.evaluation.curves import compute_pr_curve
from src.perturbations.synthetic_anomalies import inject_point_spikes, inject_sensor_freeze, inject_pattern_inversion
from src.data.download_ucr_sample import parse_ucr_filename, SAMPLE_FILENAME

DATA_FILE = os.path.join("data", "raw", SAMPLE_FILENAME)

def evaluate_scenario(scenario_name, series_to_test, y_mask, train_series):
    models = {
        "Rolling_IQR": RollingIQRDetector(window=100, k=2.5),
        "AutoRegressive_AR10": AutoRegressiveAnomalyDetector(lags=10, threshold_sigma=3.5),
        "IsolationForest": TimeSeriesIsolationForest(window_size=50, contamination=0.02, random_state=42),
        "LocalOutlierFactor": TimeSeriesLOF(window_size=50, n_neighbors=30, contamination=0.02),
    }

    results = []
    for name, detector in models.items():
        if hasattr(detector, "fit") and not isinstance(detector, RollingIQRDetector):
            detector.fit(train_series)
            pred_df = detector.predict(series_to_test)
        else:
            pred_df = detector.fit_predict(series_to_test)

        pw = evaluate_pointwise(y_mask, pred_df["anomaly_pred"].values, pred_df["score"].values)
        ev = evaluate_event_aware(y_mask, pred_df["anomaly_pred"].values)
        pr_info = compute_pr_curve(y_mask, pred_df["score"].values)

        results.append({
            "scenario": scenario_name,
            "model": name,
            "f1": pw["f1"],
            "precision": pw["precision"],
            "recall": pw["recall"],
            "total_fp": pw["total_false_alarms"],
            "event_recall": ev["event_recall"],
            "pr_auc": pr_info["pr_auc"]
        })

    return results

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} no encontrado.")
        return

    meta = parse_ucr_filename(SAMPLE_FILENAME)
    df = pd.read_csv(DATA_FILE, header=None, names=["value"])
    
    train_split = meta["train_end"]
    train_series = df["value"].iloc[:train_split].reset_index(drop=True)
    clean_test = df["value"].iloc[train_split:].reset_index(drop=True)

    print("=" * 75)
    print("DATASENTINEL: Benchmark por Morfologías de Anomalía Sintética")
    print("=" * 75)

    all_res = []

    # 1. Picos puntuales
    spikes_series, spikes_mask = inject_point_spikes(clean_test, n_spikes=30, magnitude=4.0)
    all_res.extend(evaluate_scenario("Picos_Aislados_Spikes", spikes_series, spikes_mask, train_series))

    # 2. Congelación de sensor (flatline)
    freeze_series, freeze_mask = inject_sensor_freeze(clean_test, start_idx=3000, duration=400)
    all_res.extend(evaluate_scenario("Sensor_Freeze_Flatline", freeze_series, freeze_mask, train_series))

    # 3. Inversión de dinámica temporal
    invert_series, invert_mask = inject_pattern_inversion(clean_test, start_idx=6000, duration=500)
    all_res.extend(evaluate_scenario("Inversion_De_Patron", invert_series, invert_mask, train_series))

    res_df = pd.DataFrame(all_res)

    os.makedirs(os.path.join("results", "tables"), exist_ok=True)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)

    csv_out = os.path.join("results", "tables", "benchmark_morfologias_anomalias.csv")
    res_df.to_csv(csv_out, index=False)

    print("\nRESUMEN DE RENDIMIENTO (F1-SCORE Y PR-AUC) POR TIPO DE ANOMALÍA:")
    pivot_table = res_df.pivot(index="scenario", columns="model", values=["f1", "pr_auc", "total_fp"])
    print(pivot_table)

    # Gráfica de barras comparativa
    summary_f1 = res_df.pivot(index="scenario", columns="model", values="f1")
    fig, ax = plt.subplots(figsize=(10, 5))
    summary_f1.plot(kind="bar", ax=ax, rot=15)
    ax.set_title("Sensibilidad de Modelos según Morfología de Anomalía (F1)")
    ax.set_ylabel("F1 Score")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_out = os.path.join("results", "figures", "sensibilidad_morfologias.png")
    plt.savefig(fig_out, dpi=200)
    plt.close()

    print(f"\nResultados guardados:\n- {csv_out}\n- {fig_out}")

if __name__ == "__main__":
    main()
