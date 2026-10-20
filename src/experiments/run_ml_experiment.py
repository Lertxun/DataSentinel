"""
Experimento comparativo ampliado: Baselines estadísticos vs Algoritmos de Machine Learning (iForest, LOF).
Entrena estrictamente en la región de train y evalúa en la región de test.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.detectors.baseline_statistical import RollingZScoreDetector, RollingIQRDetector
from src.detectors.isolation_forest_detector import TimeSeriesIsolationForest
from src.detectors.lof_detector import TimeSeriesLOF
from src.evaluation.metrics import evaluate_pointwise, evaluate_event_aware
from src.data.download_ucr_sample import parse_ucr_filename, SAMPLE_FILENAME

DATA_FILE = os.path.join("data", "raw", SAMPLE_FILENAME)

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} no encontrado. Ejecuta primero download_ucr_sample.py")
        return

    meta = parse_ucr_filename(SAMPLE_FILENAME)
    df = pd.read_csv(DATA_FILE, header=None, names=["value"])
    
    # Vector ground truth completo
    y_true_full = np.zeros(len(df), dtype=int)
    y_true_full[meta["anomaly_start"]:meta["anomaly_end"]] = 1
    
    # Partición temporal estricta Train / Test
    train_split = meta["train_end"]
    train_series = df["value"].iloc[:train_split].reset_index(drop=True)
    test_series = df["value"].iloc[train_split:].reset_index(drop=True)
    test_y_true = y_true_full[train_split:]
    
    print("=" * 65)
    print("DATASENTINEL EXPERIMENT: Baselines vs Machine Learning")
    print(f"Serie: {meta['series_id']} ({meta['dataset_name']})")
    print(f"Puntos Entrenamiento: {len(train_series)} | Puntos Test: {len(test_series)}")
    print(f"Puntos anómalos reales en Test: {np.sum(test_y_true)} de {len(test_series)} ({np.sum(test_y_true)/len(test_series)*100:.2f}%)")
    print("=" * 65)

    # Definir modelos a evaluar
    models = {
        "Baseline_RollingZScore_w100": {
            "type": "statistical",
            "instance": RollingZScoreDetector(window=100, threshold=3.0)
        },
        "Baseline_RollingIQR_w100": {
            "type": "statistical",
            "instance": RollingIQRDetector(window=100, k=2.5)
        },
        "ML_IsolationForest_c0.02": {
            "type": "ml",
            "instance": TimeSeriesIsolationForest(window_size=50, contamination=0.02, random_state=42)
        },
        "ML_LOF_c0.02": {
            "type": "ml",
            "instance": TimeSeriesLOF(window_size=50, n_neighbors=30, contamination=0.02)
        }
    }

    results = []
    fig, axes = plt.subplots(len(models), 1, figsize=(15, 3.5 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]

    real_anom_idx = np.where(test_y_true == 1)[0]

    for ax, (name, cfg) in zip(axes, models.items()):
        detector = cfg["instance"]
        if cfg["type"] == "ml":
            print(f"Entrenando {name} sobre partición de train...")
            detector.fit(train_series)
            pred_df = detector.predict(test_series)
        else:
            pred_df = detector.fit_predict(test_series)

        # Evaluación
        pw = evaluate_pointwise(test_y_true, pred_df["anomaly_pred"].values, pred_df["score"].values)
        ev = evaluate_event_aware(test_y_true, pred_df["anomaly_pred"].values)
        results.append({"model": name, **pw, **ev})

        # Gráfica
        ax.plot(test_series.index, test_series.values, color="#1f4e79", alpha=0.55, linewidth=0.7, label="Señal Test")
        if len(real_anom_idx) > 0:
            ax.axvspan(real_anom_idx[0], real_anom_idx[-1], color="#2ca02c", alpha=0.3, label="Anomalía Real")

        detected_idx = np.where(pred_df["anomaly_pred"] == 1)[0]
        if len(detected_idx) > 0:
            ax.scatter(detected_idx, test_series.iloc[detected_idx], color="#d62728", s=10, zorder=5, label=f"Alertas ({len(detected_idx)})")

        ax.set_title(f"{name} | F1: {pw['f1']:.3f} | Prec: {pw['precision']:.3f} | Rec: {pw['recall']:.3f} | FP: {pw['total_false_alarms']} | Event-Rec: {ev['event_recall']:.1f}", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    plt.xlabel("Índice Temporal (Test)")
    plt.tight_layout()

    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    os.makedirs(os.path.join("results", "tables"), exist_ok=True)

    fig_out = os.path.join("results", "figures", "comparativa_baselines_vs_ml.png")
    plt.savefig(fig_out, dpi=200)
    plt.close()

    res_df = pd.DataFrame(results)
    csv_out = os.path.join("results", "tables", "comparativa_baselines_vs_ml.csv")
    res_df.to_csv(csv_out, index=False)

    print("
RESULTADOS FINALES:")
    cols_to_print = ["model", "precision", "recall", "f1", "pr_auc", "total_false_alarms", "event_recall"]
    print(res_df[cols_to_print].to_string(index=False))
    print(f"
Gráfica guardada en: {fig_out}")
    print(f"Tabla guardada en: {csv_out}")

if __name__ == "__main__":
    main()
