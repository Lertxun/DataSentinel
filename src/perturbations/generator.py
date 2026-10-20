"""
Módulo de perturbaciones e inyecciones sintéticas controladas.
Permite evaluar la degradación de los detectores bajo:
1. Ruido Gaussiano aditivo (degradación SNR).
2. Concept Drift / Desplazamiento de distribución (abrupto y gradual).
3. Valores ausentes / Missing values (interpolación o ceros).
"""
import numpy as np
import pandas as pd


def inject_gaussian_noise(series: pd.Series, noise_level: float = 0.2, seed: int = 42) -> pd.Series:
    """
    Añade ruido gaussiano N(0, noise_level * std(series)).
    noise_level representa la fracción de la desviación estándar original.
    """
    np.random.seed(seed)
    std = series.std()
    noise = np.random.normal(0, noise_level * std, size=len(series))
    return pd.Series(series.values + noise, index=series.index)


def inject_concept_drift(series: pd.Series, drift_type: str = "mean_shift", magnitude: float = 1.5, start_pct: float = 0.5) -> pd.Series:
    """
    Introduce concept drift a partir de un porcentaje temporal de la serie.
    - mean_shift: desplazamiento repentino de la media (magnitude * std).
    - variance_shift: aumento o disminución de la amplitud/escala.
    - linear_trend: tendencia lineal incremental.
    """
    values = series.values.copy()
    n = len(values)
    start_idx = int(n * start_pct)
    std = np.std(values)

    if drift_type == "mean_shift":
        values[start_idx:] += magnitude * std
    elif drift_type == "variance_shift":
        mean = np.mean(values[start_idx:])
        values[start_idx:] = mean + (values[start_idx:] - mean) * magnitude
    elif drift_type == "linear_trend":
        trend = np.linspace(0, magnitude * std, n - start_idx)
        values[start_idx:] += trend
    else:
        raise ValueError(f"drift_type desconocido: {drift_type}")

    return pd.Series(values, index=series.index)
