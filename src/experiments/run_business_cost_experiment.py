"""
Experimento de Impacto en Negocio: Comparativa de Costes Asimétricos (TEC).
Evalúa cómo varía el ranking de los modelos bajo diferentes perfiles de penalización económica:
- Perfil Equilibrado (FP = 20 €, FN = 1.000 €)
- Perfil Alta Criticidad / Cero Tolerancia a Fallos (FP = 20 €, FN = 10.000 €)
- Perfil Ruido Sensible / Coste Analista Alto (FP = 150 €, FN = 1.000 €)
"""
import os
import numpy as np
import pandas as pd

from src.detectors.baseline_statistical import RollingZScoreDetector, RollingIQRDetector
from src.detectors.isolation_forest_detector import TimeSeriesIsolationForest
from src.detectors.lof_detector import TimeSeriesLOF
from src.detectors.arima_detector import AutoRegressiveAnomalyDetector
from src.evaluation.metrics import evaluate_pointwise
from src.evaluation.cost_matrix import compute_business_costs
from src.data.download_ucr_sample import parse_ucr_filename, SAMPLE_FILENAME

DATA_FILE = os.path.join("data", "raw", SAMPLE_FILENAME)

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

    print("=" * 75)
    print("DATASENTINEL: Evaluación de Impacto Económico y Detección Paramétrica (AR)")
    print("=" * 75)

    models = {
        "Rolling_IQR": ("stat", RollingIQRDetector(window=100, k=2.5)),
        "AutoRegressive_AR10": ("trainable", AutoRegressiveAnomalyDetector(lags=10, threshold_sigma=3.5)),
        "IsolationForest": ("trainable", TimeSeriesIsolationForest(window_size=50, contamination=0.02, random_state=42)),
        "LocalOutlierFactor": ("trainable", TimeSeriesLOF(window_size=50, n_neighbors=30, contamination=0.02)),
    }

    # Perfiles de coste empresarial (FP_cost, FN_cost)
    cost_profiles = {
        "Operacion_Estandar": (20.0, 1000.0),
        "Alta_Criticidad_Industrial": (20.0, 10000.0),
        "Coste_Analista_Elevado": (150.0, 1000.0)
    }

    preds = {}
    for name, (kind, detector) in models.items():
        if kind == "trainable":
            detector.fit(train_series)
            preds[name] = detector.predict(test_series)
        else:
            preds[name] = detector.fit_predict(test_series)

    all_business_rows = []

    for prof_name, (c_fp, c_fn) in cost_profiles.items():
        for model_name, pred_df in preds.items():
            pw = evaluate_pointwise(test_y_true, pred_df["anomaly_pred"].values, pred_df["score"].values)
            costs = compute_business_costs(test_y_true, pred_df["anomaly_pred"].values, cost_fp=c_fp, cost_fn=c_fn)
            all_business_rows.append({
                "perfil": prof_name,
                "modelo": model_name,
                "f1": pw["f1"],
                "precision": pw["precision"],
                "recall": pw["recall"],
                "total_fp": pw["total_false_alarms"],
                "coste_total_eur": costs["total_expected_cost_eur"],
                "ahorro_neto_eur": costs["net_savings_eur"],
                "ahorro_porcentaje": costs["savings_percentage"]
            })

    business_df = pd.DataFrame(all_business_rows)
    
    os.makedirs(os.path.join("results", "tables"), exist_ok=True)
    csv_out = os.path.join("results", "tables", "evaluacion_costes_negocio.csv")
    business_df.to_csv(csv_out, index=False)

    print("\nRESUMEN DE COSTE TOTAL ESPERADO Y AHORRO POR PERFIL:\n")
    for prof in cost_profiles.keys():
        sub = business_df[business_df["perfil"] == prof]
        print(f"--- Escenario: {prof} (Coste FP: {cost_profiles[prof][0]}€ | Coste FN: {cost_profiles[prof][1]}€) ---")
        cols = ["modelo", "f1", "recall", "total_fp", "coste_total_eur", "ahorro_porcentaje"]
        print(sub[cols].sort_values(by="coste_total_eur").to_string(index=False))
        print()

    print(f"Resultados completos guardados en: {csv_out}")

if __name__ == "__main__":
    main()
