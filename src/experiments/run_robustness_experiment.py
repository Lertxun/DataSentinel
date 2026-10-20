"""
Fase 3: Experimento de Robustez bajo Perturbaciones Controladas.
Evalúa cómo se degradan F1, Precision, Recall y las Falsas Alarmas (FP)
cuando inyectamos diferentes niveles de Ruido Gaussiano y Concept Drift.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.detectors.baseline_statistical import RollingZScoreDetector, RollingIQRDetector
from src.detectors.isolation_forest_detector import TimeSeriesIsolationForest
from src.detectors.lof_detector import TimeSeriesLOF
from src.evaluation.metrics import evaluate_pointwise, evaluate_event_aware
from src.perturbations.generator import inject_gaussian_noise, inject_concept_drift
from src.data.download_ucr_sample import parse_ucr_filename, SAMPLE_FILENAME

DATA_FILE = os.path.join("data", "raw", SAMPLE_FILENAME)

def run_evaluation(model_name, detector, is_ml, train_series, test_series, y_true):
    if is_ml:
        detector.fit(train_series)
        pred_df = detector.predict(test_series)
    else:
        pred_df = detector.fit_predict(test_series)
        
    pw = evaluate_pointwise(y_true, pred_df["anomaly_pred"].values, pred_df["score"].values)
    ev = evaluate_event_aware(y_true, pred_df["anomaly_pred"].values)
    return {"model": model_name, **pw, **ev}

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} no encontrado.")
        return

    meta = parse_ucr_filename(SAMPLE_FILENAME)
    df = pd.read_csv(DATA_FILE, header=None, names=["value"])
    
    y_true_full = np.zeros(len(df), dtype=int)
    y_true_full[meta["anomaly_start"]:meta["anomaly_end"]] = 1
    
    train_split = meta["train_end"]
    train_series = df["value"].iloc[:train_split].reset_index(drop=True)
    test_series = df["value"].iloc[train_split:].reset_index(drop=True)
    test_y_true = y_true_full[train_split:]
    
    # 1. Batería de Niveles de Ruido
    noise_levels = [0.0, 0.1, 0.25, 0.5]
    
    print("=" * 65)
    print("DATASENTINEL FASE 3: Batería de Robustez frente a Ruido")
    print("=" * 65)

    all_noise_results = []
    
    for nl in noise_levels:
        print(f"\n--> Evaluando con nivel de ruido = {nl}...")
        noisy_test = inject_gaussian_noise(test_series, noise_level=nl, seed=42) if nl > 0.0 else test_series
        
        models = [
            ("RollingZScore", RollingZScoreDetector(window=100, threshold=3.0), False),
            ("RollingIQR", RollingIQRDetector(window=100, k=2.5), False),
            ("IsolationForest", TimeSeriesIsolationForest(window_size=50, contamination=0.02, random_state=42), True),
            ("LOF", TimeSeriesLOF(window_size=50, n_neighbors=30, contamination=0.02), True),
        ]
        
        for name, detector, is_ml in models:
            res = run_evaluation(name, detector, is_ml, train_series, noisy_test, test_y_true)
            res["noise_level"] = nl
            all_noise_results.append(res)
            
    noise_df = pd.DataFrame(all_noise_results)
    
    # Guardar resultados
    os.makedirs(os.path.join("results", "tables"), exist_ok=True)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    
    csv_path = os.path.join("results", "tables", "robustez_ruido_metrics.csv")
    noise_df.to_csv(csv_path, index=False)
    
    # Gráfica de degradación: F1 vs Ruido y Falsas Alarmas vs Ruido
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_name in noise_df["model"].unique():
        sub = noise_df[noise_df["model"] == model_name]
        ax1.plot(sub["noise_level"], sub["f1"], marker="o", label=model_name)
        ax2.plot(sub["noise_level"], sub["total_false_alarms"], marker="s", label=model_name)
        
    ax1.set_title("Degradación de F1 según Nivel de Ruido")
    ax1.set_xlabel("Nivel de Ruido (fracción de sigma)")
    ax1.set_ylabel("F1 Score")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title("Explosión de Falsas Alarmas (FP) según Ruido")
    ax2.set_xlabel("Nivel de Ruido (fracción de sigma)")
    ax2.set_ylabel("Total Falsas Alarmas")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig_path = os.path.join("results", "figures", "robustez_ruido_degradacion.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    
    print("\nRESUMEN DE ROBUSTEZ FRENTE A RUIDO (F1 Score y Falsas Alarmas):")
    summary = noise_df.pivot(index="noise_level", columns="model", values=["f1", "total_false_alarms"])
    print(summary)
    print(f"\nResultados guardados en:\n- {csv_path}\n- {fig_path}")

if __name__ == "__main__":
    main()
