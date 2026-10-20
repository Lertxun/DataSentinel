"""
Detector de anomalías basado en Local Outlier Factor (LOF).
Utiliza novedad (novelty=True) para entrenar con datos de referencia y predecir en test.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from src.preprocessing.sliding_window import extract_window_features


class TimeSeriesLOF:
    def __init__(self, window_size: int = 50, n_neighbors: int = 20, contamination: float = 0.01):
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, train_series: pd.Series):
        features_df = extract_window_features(train_series, window_size=self.window_size)
        # Submuestreo si train es muy grande para mantener ejecución rápida en local
        if len(features_df) > 10000:
            sample_df = features_df.sample(n=10000, random_state=42)
            self.model.fit(sample_df.values)
        else:
            self.model.fit(features_df.values)
        self.is_fitted = True
        return self

    def predict(self, test_series: pd.Series) -> pd.DataFrame:
        features_df = extract_window_features(test_series, window_size=self.window_size)
        raw_scores = -self.model.score_samples(features_df.values)
        raw_preds = self.model.predict(features_df.values)
        binary_preds = (raw_preds == -1).astype(int)

        return pd.DataFrame({
            "value": test_series,
            "score": raw_scores,
            "anomaly_pred": binary_preds
        }, index=test_series.index)
