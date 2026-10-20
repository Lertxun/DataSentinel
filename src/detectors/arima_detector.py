"""
Detector de anomalías basado en Residuos de Modelos Temporales Autorregresivos (AR / ARIMA).
Aprende la dinámica de la señal normal durante el entrenamiento y predice paso a paso;
las discrepancias severas (|y_t - y_hat_t| > threshold * sigma_e) se catalogan como anomalías.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


class AutoRegressiveAnomalyDetector:
    """
    Detector basado en autorregresión estricta sobre la señal de referencia.
    Calcula el error cuadrático / absoluto de predicción causal.
    """
    def __init__(self, lags: int = 10, threshold_sigma: float = 3.5):
        self.lags = lags
        self.threshold_sigma = threshold_sigma
        self.model_fitted = None
        self.residual_std = 1.0

    def fit(self, train_series: pd.Series):
        # Ajustamos un modelo autorregresivo AR(p) sobre los datos normales
        model = AutoReg(train_series.values, lags=self.lags, old_names=False)
        self.model_fitted = model.fit()
        
        # Desviación típica de los residuos en entrenamiento
        residuals_train = self.model_fitted.resid
        self.residual_std = float(np.std(residuals_train)) if len(residuals_train) > 0 else 1.0
        if self.residual_std == 0:
            self.residual_std = 1e-6
        return self

    def predict(self, test_series: pd.Series) -> pd.DataFrame:
        """
        Calcula el residuo causal para cada punto en test:
        y_hat_t = c + sum(phi_i * y_{t-i})
        e_t = |y_t - y_hat_t| / sigma_e
        """
        values = test_series.values
        n = len(values)
        scores = np.zeros(n)
        
        params = self.model_fitted.params
        intercept = params[0]
        phi = params[1:]
        p = len(phi)

        # Para los primeros p puntos usamos la media
        scores[:p] = 0.0

        for t in range(p, n):
            past_window = values[t - p : t][::-1]  # [y_{t-1}, y_{t-2}, ..., y_{t-p}]
            y_pred = intercept + np.dot(phi, past_window)
            err = np.abs(values[t] - y_pred)
            scores[t] = err / self.residual_std

        scores_series = pd.Series(scores, index=test_series.index)
        anomalies = (scores_series >= self.threshold_sigma).astype(int)

        return pd.DataFrame({
            "value": test_series,
            "score": scores_series,
            "anomaly_pred": anomalies
        }, index=test_series.index)
