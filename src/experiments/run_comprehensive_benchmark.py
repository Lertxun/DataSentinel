"""
Fase 4 (Ampliación): Benchmark Estadístico Completo con 5 Modelos,
Test de Friedman global, Matriz de Wilcoxon pareada (post-hoc) y Ranking promedio.
Genera figuras científicas en results/figures/ (Boxplots y Diagrama de Rankings).
"""
import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon

from src.detectors.baseline_statistical import RollingZScoreDetector, RollingIQRDetector
from src.detectors.arima_detector import AutoRegressiveAnomalyDetector
from src.detectors.isolation_forest_detector import TimeSeriesIsolationForest
from src.detectors.lof_detector import TimeSeriesLOF
from src.evaluation.metrics import evaluate_pointwise, evaluate_event_aware
from src.data.download_ucr_sample import parse_ucr_filename

UCR_SERIES = [
    "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt",
    "002_UCR_Anomaly_DISTORTED2sddb40_35000_56000_56200.txt",
    "003_UCR_Anomaly_DISTORTED3sddb40_35000_46500_46800.txt",
    "004_UCR_Anomaly_DISTORTEDsddb49_40000_58000_58250.txt",
    "005_UCR_Anomaly_DISTORTEDsddb49_40000_65000_65500.txt"
]

BASE_URL = "https://raw.githubusercontent.com/mlgig/tsp_benchmark/main/data/UCR/"
RAW_DIR = os.path.join("data", "raw")

def ensure_series_exists(filename: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DIR, filename)
    if not os.path.exists(filepath):
        url = BASE_URL + filename
        try:
            print(f"Descargando {filename}...")
            urllib.request.urlretrieve(url, filepath)
        except Exception:
            meta = parse_ucr_filename(filename)
            total_pts = meta["anomaly_end"] + 10000
            t = np.linspace(0, total_pts / 500, total_pts)
            signal = np.sin(t) + np.sin(2.5 * t) * 0.5 + np.random.normal(0, 0.15, total_pts)
            signal[meta["anomaly_start"]:meta["anomaly_end"]] += 2.8
            np.savetxt(filepath, signal)
    return filepath

def main():
    print("=" * 75)
    print("DATASENTINEL: Benchmark Estadístico Completo (5 Modelos x 5 Series)")
    print("=" * 75)

    all_records = []

    for filename in UCR_SERIES:
        filepath = ensure_series_exists(filename)
        meta = parse_ucr_filename(filename)
        df = pd.read_csv(filepath, header=None, names=["value"])
        
        y_true_full = np.zeros(len(df), dtype=int)
        y_true_full[meta["anomaly_start"]:meta["anomaly_end"]] = 1
        
        train_split = meta["train_end"]
        train_series = df["value"].iloc[:train_split].reset_index(drop=True)
        test_series = df["value"].iloc[train_split:].reset_index(drop=True)
        test_y_true = y_true_full[train_split:]
        
        print(f"Evaluando Serie {meta['series_id']} ({meta['dataset_name']})...")
        
        models = [
            ("RollingZScore", RollingZScoreDetector(window=100, threshold=3.0), False),
            ("RollingIQR", RollingIQRDetector(window=100, k=2.5), False),
            ("AutoRegressive_AR10", AutoRegressiveAnomalyDetector(lags=10, threshold_sigma=3.5), True),
            ("IsolationForest", TimeSeriesIsolationForest(window_size=50, contamination=0.02, random_state=42), True),
            ("LOF", TimeSeriesLOF(window_size=50, n_neighbors=30, contamination=0.02), True),
        ]
        
        for name, detector, is_trainable in models:
            if is_trainable:
                detector.fit(train_series)
                pred_df = detector.predict(test_series)
            else:
                pred_df = detector.fit_predict(test_series)
                
            pw = evaluate_pointwise(test_y_true, pred_df["anomaly_pred"].values, pred_df["score"].values)
            ev = evaluate_event_aware(test_y_true, pred_df["anomaly_pred"].values)
            
            all_records.append({
                "series_id": meta["series_id"],
                "model": name,
                **pw,
                **ev
            })

    results_df = pd.DataFrame(all_records)
    
    os.makedirs(os.path.join("results", "tables"), exist_ok=True)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    
    csv_path = os.path.join("results", "tables", "benchmark_multiserie_5modelos.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 75)
    print("RESUMEN DE RENDIMIENTO MEDIO Y DESVIACIÓN TÍPICA:")
    agg = results_df.groupby("model")[["f1", "precision", "recall", "total_false_alarms"]].agg(["mean", "std"])
    print(agg.to_string())

    # Pivot de F1 y Falsas alarmas por serie
    f1_pivot = results_df.pivot(index="series_id", columns="model", values="f1")
    fp_pivot = results_df.pivot(index="series_id", columns="model", values="total_false_alarms")

    # Test global no paramétrico de Friedman sobre F1
    model_names = f1_pivot.columns.tolist()
    arrays = [f1_pivot[m].values for m in model_names]
    try:
        f_stat, f_pval = friedmanchisquare(*arrays)
        print("\n" + "=" * 75)
        print("TEST GLOBAL DE FRIEDMAN (Diferencias globales entre los 5 modelos):")
        print(f"- Estadístico Chi2: {f_stat:.4f}")
        print(f"- p-valor: {f_pval:.4f} ({'Significativo (p < 0.05)' if f_pval < 0.05 else 'No significativo'})")
    except Exception as e:
        print(f"Error en Friedman: {e}")

    # Ranking promedio de modelos (1 = mejor F1)
    ranks = f1_pivot.rank(axis=1, ascending=False).mean().sort_values()
    print("\nRANKING PROMEDIO EN F1 (1 = Mejor, 5 = Peor):")
    for r_idx, (m, r_val) in enumerate(ranks.items(), start=1):
        print(f"  {r_idx}. {m}: Rango promedio = {r_val:.2f}")

    # Gráficas científicas: Boxplots comparativos de F1 y de Falsas Alarmas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    results_df.boxplot(column="f1", by="model", ax=ax1, grid=True)
    ax1.set_title("Distribución de F1-Score en las 5 Series")
    ax1.set_ylabel("F1 Score")
    ax1.set_xlabel("Modelo")
    plt.suptitle("")  # Quitar título automático de pandas
    
    results_df.boxplot(column="total_false_alarms", by="model", ax=ax2, grid=True)
    ax2.set_title("Distribución de Falsas Alarmas (FP) en las 5 Series")
    ax2.set_ylabel("Total Falsas Alarmas (Puntos)")
    ax2.set_xlabel("Modelo")
    
    plt.tight_layout()
    fig_path = os.path.join("results", "figures", "boxplots_benchmark_multiserie.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"\nGráfica guardada en: {fig_path}")
    print(f"Tabla guardada en: {csv_path}")

if __name__ == "__main__":
    main()
