"""
Detectores estadísticos de referencia (Baselines).
Implementa Rolling Z-Score y Rolling IQR adaptados a series temporales
sin fuga de información hacia el pasado (look-ahead bias).
"""
import numpy as np
import pandas as pd


class RollingZScoreDetector:
    """
    Detector basado en media y desviación típica móviles (Rolling Z-score).
    
    Parámetros:
    - window: tamaño de la ventana móvil hacia el pasado.
    - threshold: umbral de desviaciones estándar para disparar anomalía (p. ej. 3.0).
    - min_periods: número mínimo de observaciones para calcular estadísticas.
    """
    def __init__(self, window: int = 50, threshold: float = 3.0, min_periods: int = 10):
        self.window = window
        self.threshold = threshold
        self.min_periods = min_periods

    def fit_predict(self, series: pd.Series) -> pd.DataFrame:
        """
        Calcula las puntuaciones de anomalía y predice etiquetas binarias.
        Evita look-ahead bias calculando la media y std sobre observaciones pasadas (shift(1)).
        """
        # Desplazamos 1 para que el punto actual x_t no contamine su propia media/std de referencia
        rolling_mean = series.shift(1).rolling(window=self.window, min_periods=self.min_periods).mean()
        rolling_std = series.shift(1).rolling(window=self.window, min_periods=self.min_periods).std()
        
        # Evitar división por cero
        rolling_std = rolling_std.replace(0, 1e-6).fillna(1.0)
        rolling_mean = rolling_mean.bfill().fillna(series.mean())
        
        # Anomaly score: valor absoluto del Z-score
        scores = np.abs((series - rolling_mean) / rolling_std)
        scores = scores.fillna(0.0)
        
        # Predicción binaria (1 = Anomalía, 0 = Normal)
        predictions = (scores >= self.threshold).astype(int)
        
        return pd.DataFrame({
            "value": series,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "score": scores,
            "anomaly_pred": predictions
        }, index=series.index)


class RollingIQRDetector:
    """
    Detector robusto basado en Rango Intercuartílico móvil (IQR).
    Menos sensible a outliers previos dentro de la ventana que el Z-Score clásico.
    """
    def __init__(self, window: int = 50, k: float = 1.5, min_periods: int = 10):
        self.window = window
        self.k = k
        self.min_periods = min_periods

    def fit_predict(self, series: pd.Series) -> pd.DataFrame:
        past = series.shift(1)
        q25 = past.rolling(window=self.window, min_periods=self.min_periods).quantile(0.25)
        q75 = past.rolling(window=self.window, min_periods=self.min_periods).quantile(0.75)
        iqr = q75 - q25
        
        lower_bound = q25 - (self.k * iqr)
        upper_bound = q75 + (self.k * iqr)
        
        lower_bound = lower_bound.bfill()
        upper_bound = upper_bound.bfill()
        
        is_anomaly = ((series < lower_bound) | (series > upper_bound)).astype(int)
        
        # Distancia normalizada respecto al margen más cercano como score
        score = np.maximum(
            (lower_bound - series) / (iqr + 1e-6),
            (series - upper_bound) / (iqr + 1e-6)
        )
        score = np.maximum(0.0, score.fillna(0.0))
        
        return pd.DataFrame({
            "value": series,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "score": score,
            "anomaly_pred": is_anomaly
        }, index=series.index)
