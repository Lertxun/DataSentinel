"""
Generador de morfologías de anomalías controladas.
Permite clasificar y evaluar la sensibilidad de los detectores ante fallos específicos:
1. Anomalías Puntuales (Spikes / Picos aislados).
2. Anomalías Colectivas de Congelación (Sensor Freeze / Flatline).
3. Anomalías Contextuales / Patrón (Pérdida de periodicidad o inversión de señal).
4. Anomalías de Varianza (Subida o caída súbita de micro-fluctuación).
"""
import numpy as np
import pandas as pd


def inject_point_spikes(series: pd.Series, n_spikes: int = 20, magnitude: float = 3.5, seed: int = 42) -> tuple:
    """
    Inyecta picos aislados (outliers puntuales) en posiciones aleatorias de la serie.
    Retorna: (serie_modificada, mascara_binaria_anomalias)
    """
    np.random.seed(seed)
    values = series.values.copy()
    n = len(values)
    mask = np.zeros(n, dtype=int)
    
    std = np.std(values)
    spike_indices = np.random.choice(np.arange(100, n - 100), size=n_spikes, replace=False)
    
    for idx in spike_indices:
        sign = np.random.choice([-1, 1])
        values[idx] += sign * magnitude * std
        mask[idx] = 1
        
    return pd.Series(values, index=series.index), mask


def inject_sensor_freeze(series: pd.Series, start_idx: int = 2000, duration: int = 300) -> tuple:
    """
    Simula una anomalía colectiva por fallo de sensor congelado (flatline / valor estancado).
    Muy común en telemetría industrial cuando un transmisor se bloquea.
    """
    values = series.values.copy()
    n = len(values)
    end_idx = min(start_idx + duration, n)
    mask = np.zeros(n, dtype=int)
    
    freeze_val = values[start_idx]
    values[start_idx:end_idx] = freeze_val
    mask[start_idx:end_idx] = 1
    
    return pd.Series(values, index=series.index), mask


def inject_pattern_inversion(series: pd.Series, start_idx: int = 5000, duration: int = 500) -> tuple:
    """
    Invierte la señal respecto a su media local.
    Mantiene la misma escala y media pero destruye la dinámica temporal aprendida.
    """
    values = series.values.copy()
    n = len(values)
    end_idx = min(start_idx + duration, n)
    mask = np.zeros(n, dtype=int)
    
    local_mean = np.mean(values[start_idx:end_idx])
    values[start_idx:end_idx] = local_mean - (values[start_idx:end_idx] - local_mean)
    mask[start_idx:end_idx] = 1
    
    return pd.Series(values, index=series.index), mask
