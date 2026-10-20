"""
Preprocesamiento: Extracción de subsecuencias y características mediante ventanas deslizantes.
Permite transformar una serie 1D en matrices multidimensionales para algoritmos de ML
(Isolation Forest, LOF, etc.) sin fuga de información futura.
"""
import numpy as np
import pandas as pd


def create_sliding_windows(series: pd.Series, window_size: int = 50, step: int = 1) -> tuple:
    """
    Construye una matriz de ventanas deslizantes (raw subsequences).
    Cada fila i contiene los valores de la serie desde t - window_size + 1 hasta t.
    Retorna: (X, indices_temporales)
    """
    values = series.values
    n = len(values)
    if n < window_size:
        raise ValueError(f"La serie tiene longitud {n}, menor que el tamaño de ventana {window_size}")

    windows = []
    indices = []
    for i in range(window_size - 1, n, step):
        window = values[i - window_size + 1 : i + 1]
        windows.append(window)
        indices.append(i)

    return np.array(windows), np.array(indices)


def extract_window_features(series: pd.Series, window_size: int = 50) -> pd.DataFrame:
    """
    Extrae un vector de características estadísticas temporales sobre ventanas causales pasadas:
    - Media, desviación típica, asimetría (skew), curtosis, min, max, diferencia pico a pico y pendiente.
    Garantiza causalidad desplazando 1 posición con shift(1).
    """
    past = series.shift(1)
    
    rolling = past.rolling(window=window_size, min_periods=window_size // 2)
    mean = rolling.mean().bfill()
    std = rolling.std().replace(0, 1e-6).bfill()
    min_val = rolling.min().bfill()
    max_val = rolling.max().bfill()
    ptp = max_val - min_val
    
    # Diferencia instantánea respecto a la media móvil
    diff_mean = (series - mean) / std
    
    features = pd.DataFrame({
        "raw_value": series,
        "rolling_mean": mean,
        "rolling_std": std,
        "peak_to_peak": ptp,
        "diff_from_mean": diff_mean,
    }, index=series.index)
    
    return features.fillna(0.0)
