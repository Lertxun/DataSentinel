"""
Script de descarga y visualización de una serie de muestra del UCR Anomaly Archive.
Nomenclatura UCR:
{ID}_{dataset}_{training_split}_{anomaly_start}_{anomaly_end}.txt
"""
import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Serie de muestra representativa (ECG / señal biomédica o acelerómetro)
SAMPLE_FILENAME = "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
BASE_URL = "https://raw.githubusercontent.com/mlgig/tsp_benchmark/main/data/UCR/"
OUTPUT_DIR = "data/raw"

def parse_ucr_filename(filename: str):
    """Extrae metadatos a partir del nombre del fichero estándar de UCR."""
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split("_")
    
    anomaly_end = int(parts[-1])
    anomaly_start = int(parts[-2])
    train_end = int(parts[-3])
    dataset_name = "_".join(parts[1:-3])
    series_id = parts[0]
    
    return {
        "series_id": series_id,
        "dataset_name": dataset_name,
        "train_end": train_end,
        "anomaly_start": anomaly_start,
        "anomaly_end": anomaly_end,
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, SAMPLE_FILENAME)
    
    if not os.path.exists(file_path):
        url = BASE_URL + SAMPLE_FILENAME
        print(f"Descargando serie de prueba desde: {url}...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print("Descarga completada con éxito.")
        except Exception as e:
            print(f"Aviso: descarga directa no disponible ({e}). Generando serie sintética de sustitución para prueba local...")
            # Fallback a serie local si no hay conexión
            n_points = 60000
            t = np.linspace(0, 100, n_points)
            signal = np.sin(t) + np.random.normal(0, 0.1, n_points)
            signal[52000:52620] += 3.5
            np.savetxt(file_path, signal)
    
    meta = parse_ucr_filename(SAMPLE_FILENAME)
    data = pd.read_csv(file_path, header=None, names=["value"])
    
    print("\nInformación de la serie:")
    print(f"- Filas totales: {len(data)}")
    print(f"- Límite región entrenamiento: {meta['train_end']}")
    print(f"- Anomalía etiquetada: [{meta['anomaly_start']}, {meta['anomaly_end']}]")

    # Gráfica exploratoria
    plt.figure(figsize=(14, 5))
    plt.plot(data.index, data["value"], label="Señal", color="#1f77b4", linewidth=0.8)
    plt.axvline(x=meta["train_end"], color="gray", linestyle="--", label="Fin de Entrenamiento")
    plt.axvspan(meta["anomaly_start"], meta["anomaly_end"], color="red", alpha=0.35, label="Anomalía Etiquetada")
    
    plt.title(f"UCR Anomaly Archive - Serie {meta['series_id']} ({meta['dataset_name']})")
    plt.xlabel("Índice Temporal")
    plt.ylabel("Valor")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/exploracion_ucr_sample.png", dpi=200, bbox_inches="tight")
    print("Gráfica guardada en: results/figures/exploracion_ucr_sample.png")

if __name__ == "__main__":
    main()
