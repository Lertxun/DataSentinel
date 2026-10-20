"""
Fase 3 (Extensión): Experimento de Concept Drift (Deriva de Concepto).
Evalúa la degradación de los detectores cuando la señal sufre:
1. Desplazamiento abrupto de la media (Mean Shift).
2. Cambio en la escala/varianza (Variance Shift).
3. Tendencia lineal acumulada (Linear Trend).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.detectors.baseline_statistical import RollingZScoreDetector, RollingIQRDetector
from src.detectors.isolation_forest_detector import TimeSeriesIsolationForest
from src.detectors.lof_detector import TimeSeriesLOF
from src.evaluation.metrics import evaluate_pointwise, evaluate_event_aware
from src.perturbations.generator import inject_concept_drift
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
    
    drift_scenarios = [
        ("Base (Sin Drift)", None, 0.0),
        ("Mean Shift (+1.5 sigma)", "mean_shift", 1.5),
        ("Mean Shift (+3.0 sigma)", "mean_shift", 3.0),
        ("Variance Shift (x2.0)", "variance_shift", 2.0),
        ("Linear Trend (+2.0 sigma)", "linear_trend", 2.0),
    ]

    print("=" * 65)
    print("DATASENTINEL FASE 3: Batería de Robustez frente a Concept Drift")
    print("=" * 65)

    all_results = []
    
    for scenario_name, dtype, mag in drift_scenarios:
        print(f"\n--> Evaluando escenario: {scenario_name}...")
        if dtype is not None:
            drifted_test = inject_concept_drift(test_series, drift_type=dtype, magnitude=mag, start_pct=0.3)
        else:
            drifted_test = test_series
            
        models = [
            ("RollingZScore", RollingZScoreDetector(window=100, threshold=3.0), False),
            ("RollingIQR", RollingIQRDetector(window=100, k=2.5), False),
            ("IsolationForest", TimeSeriesIsolationForest(window_size=50, contamination=0.02, random_state=42), True),
            ("LOF", TimeSeriesLOF(window_size=50, n_neighbors=30, contamination=0.02), True),
        ]
        
        for name, detector, is_ml in models:
            res = run_evaluation(name, detector, is_ml, train_series, drifted_test, test_y_true)
            res["scenario"] = scenario_name
            all_results.append(res)
            
    drift_df = pd.DataFrame(all_results)
    
    os.makedirs(os.path.join("results", "tables"), exist_ok=True)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    
    csv_path = os.path.join("results", "tables", "robustez_drift_metrics.csv")
    drift_df.to_csv(csv_path, index=False)
    
    # Gráfica de barras comparativa de F1 y Falsas Alarmas por escenario
    summary_f1 = drift_df.pivot(index="scenario", columns="model", values="f1")
    summary_fp = drift_df.pivot(index="scenario", columns="model", values="total_false_alarms")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    summary_f1.plot(kind="bar", ax=ax1, rot=25, title="F1-Score bajo Concept Drift")
    ax1.set_ylabel("F1")
    ax1.grid(True, alpha=0.3)
    
    summary_fp.plot(kind="bar", ax=ax2, rot=25, title="Falsas Alarmas (FP) bajo Concept Drift")
    ax2.set_ylabel("Total FP")
    ax2.grid(True, alpha=0.3)
    
    fig_path = os.path.join("results", "figures", "robustez_drift_comparativa.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    
    print("\nRESUMEN DE CONCEPT DRIFT (F1 Score):")
    print(summary_f1)
    print("\nRESUMEN DE FALSAS ALARMAS (FP):")
    print(summary_fp)
    print(f"\nResultados guardados en:\n- {csv_path}\n- {fig_path}")

if __name__ == "__main__":
    main()
