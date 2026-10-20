"""
Ejecutor de experimentos para comparar detectores baselines sobre la serie UCR.
Guarda métricas en results/tables y figuras en results/figures.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.detectors.baseline_statistical import RollingZScoreDetector, RollingIQRDetector
from src.evaluation.metrics import evaluate_pointwise, evaluate_event_aware
from src.data.download_ucr_sample import parse_ucr_filename, SAMPLE_FILENAME

DATA_FILE = os.path.join("data/raw", SAMPLE_FILENAME)

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} no existe. Ejecuta primero download_ucr_sample.py")
        return

    meta = parse_ucr_filename(SAMPLE_FILENAME)
    df = pd.read_csv(DATA_FILE, header=None, names=["value"])
    
    # Crear vector de ground truth (y_true)
    y_true = np.zeros(len(df), dtype=int)
    y_true[meta["anomaly_start"]:meta["anomaly_end"]] = 1
    
    # Solo evaluamos en la partición de test (a partir de train_end) para simular vida real
    test_idx = np.arange(meta["train_end"], len(df))
    test_series = df["value"].iloc[test_idx].reset_index(drop=True)
    test_y_true = y_true[test_idx]
    
    print(f"Evaluando sobre partición de test: {len(test_series)} observaciones.")
    print(f"Anomalías reales en test: {np.sum(test_y_true)} puntos.")

    detectors = {
        "Rolling_ZScore_w50_t3": RollingZScoreDetector(window=50, threshold=3.0),
        "Rolling_ZScore_w100_t3": RollingZScoreDetector(window=100, threshold=3.0),
        "Rolling_IQR_w100_k1.5": RollingIQRDetector(window=100, k=1.5),
        "Rolling_IQR_w100_k3.0": RollingIQRDetector(window=100, k=3.0),
    }

    results = []
    
    fig, axes = plt.subplots(len(detectors), 1, figsize=(14, 3.5 * len(detectors)), sharex=True)
    if len(detectors) == 1:
        axes = [axes]

    for ax, (name, detector) in zip(axes, detectors.items()):
        pred_df = detector.fit_predict(test_series)
        
        # Métricas
        pw = evaluate_pointwise(test_y_true, pred_df["anomaly_pred"].values, pred_df["score"].values)
        ev = evaluate_event_aware(test_y_true, pred_df["anomaly_pred"].values)
        
        row = {"model": name, **pw, **ev}
        results.append(row)
        
        # Graficar predicción
        ax.plot(test_series.index, test_series.values, label="Señal", color="#2b5c8f", alpha=0.6, linewidth=0.8)
        
        # Marcar anomalía real
        real_anomaly_indices = np.where(test_y_true == 1)[0]
        if len(real_anomaly_indices) > 0:
            ax.axvspan(real_anomaly_indices[0], real_anomaly_indices[-1], color="green", alpha=0.25, label="Anomalía Real")
            
        # Puntos detectados como anomalía
        detected_indices = np.where(pred_df["anomaly_pred"] == 1)[0]
        if len(detected_indices) > 0:
            ax.scatter(detected_indices, test_series.iloc[detected_indices], color="red", s=8, zorder=5, label=f"Alertas ({len(detected_indices)})")
            
        ax.set_title(f"{name} | F1: {pw['f1']:.3f} | Prec: {pw['precision']:.3f} | Rec: {pw['recall']:.3f} | FP: {pw['total_false_alarms']}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plt.xlabel("Índice Temporal (Test)")
    plt.tight_layout()
    
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    
    fig_path = "results/figures/comparativa_baselines_test.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    
    results_df = pd.DataFrame(results)
    csv_path = "results/tables/baselines_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"\nResultados guardados:")
    print(f"- Gráfica: {fig_path}")
    print(f"- Tabla CSV: {csv_path}")
    print("\nResumen de métricas:")
    print(results_df[["model", "precision", "recall", "f1", "total_false_alarms", "event_recall"]].to_string(index=False))

if __name__ == "__main__":
    main()
