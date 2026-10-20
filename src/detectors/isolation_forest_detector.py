"""
Detector de anomalías basado en Isolation Forest (iForest).
Aplica aislamiento sobre subespacios de características temporales extraídas de la serie.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.preprocessing.sliding_window import extract_window_features


class TimeSeriesIsolationForest:
    """
    Isolation Forest adaptado a series temporales.
    Aprende la distribución normal sobre los datos de entrenamiento (train split)
    y evalúa observaciones de prueba (test split).
    """
    def __init__(self, window_size: int = 50, contamination: float = 0.01, n_estimators: int = 100, random_state: int = 42):
        self.window_size = window_size
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, train_series: pd.Series):
        """Entrena el bosque únicamente sobre la partición normal o de entrenamiento."""
        features_df = extract_window_features(train_series, window_size=self.window_size)
        self.model.fit(features_df.values)
        self.is_fitted = True
        return self

    def predict(self, test_series: pd.Series) -> pd.DataFrame:
        """
        Calcula el anomaly score y la etiqueta binaria en la partición de prueba.
        En scikit-learn:
        - score_samples devuelve valores más negativos cuanto más anómalo.
          Invertimos el signo para que mayor score = más anómalo.
        """
        features_df = extract_window_features(test_series, window_size=self.window_size)
        raw_scores = -self.model.score_samples(features_df.values)
        
        # Predicción (-1 anómalo, 1 normal en scikit-learn -> convertimos a 1 anómalo, 0 normal)
        raw_preds = self.model.predict(features_df.values)
        binary_preds = (raw_preds == -1).astype(int)

        return pd.DataFrame({
            "value": test_series,
            "score": raw_scores,
            "anomaly_pred": binary_preds
        }, index=test_series.index)
