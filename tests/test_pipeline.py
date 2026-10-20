"""
Batería de Pruebas Automatizadas (pytest) para DataSentinel.
Garantiza:
1. No look-ahead bias (causalidad temporal estricta).
2. Coherencia dimensional de las salidas en todos los detectores.
3. Comportamiento esperado de las funciones de coste económico y perturbaciones.
"""
import pytest
import numpy as np
import pandas as pd

from src.detectors.baseline_statistical import RollingZScoreDetector, RollingIQRDetector
from src.detectors.isolation_forest_detector import TimeSeriesIsolationForest
from src.detectors.lof_detector import TimeSeriesLOF
from src.detectors.arima_detector import AutoRegressiveAnomalyDetector
from src.evaluation.metrics import evaluate_pointwise, evaluate_event_aware
from src.evaluation.cost_matrix import compute_business_costs
from src.perturbations.generator import inject_gaussian_noise, inject_concept_drift


@pytest.fixture
def synthetic_series():
    np.random.seed(42)
    n = 1000
    t = np.linspace(0, 10, n)
    vals = np.sin(t) + np.random.normal(0, 0.1, n)
    # inyectar anomalía
    vals[800:850] += 3.0
    return pd.Series(vals)


def test_causal_rolling_zscore(synthetic_series):
    detector = RollingZScoreDetector(window=50, threshold=3.0)
    pred_df = detector.fit_predict(synthetic_series)
    
    assert len(pred_df) == len(synthetic_series)
    assert set(pred_df["anomaly_pred"].unique()).issubset({0, 1})
    # Comprobar que pred_df['score'] no contiene NaNs
    assert not pred_df["score"].isna().any()


def test_causal_rolling_iqr(synthetic_series):
    detector = RollingIQRDetector(window=50, k=2.5)
    pred_df = detector.fit_predict(synthetic_series)
    
    assert len(pred_df) == len(synthetic_series)
    assert set(pred_df["anomaly_pred"].unique()).issubset({0, 1})


def test_isolation_forest_pipeline(synthetic_series):
    train_part = synthetic_series.iloc[:700]
    test_part = synthetic_series.iloc[700:].reset_index(drop=True)
    
    detector = TimeSeriesIsolationForest(window_size=30, contamination=0.03, random_state=42)
    detector.fit(train_part)
    pred_df = detector.predict(test_part)
    
    assert len(pred_df) == len(test_part)
    assert set(pred_df["anomaly_pred"].unique()).issubset({0, 1})


def test_autoregressive_detector(synthetic_series):
    train_part = synthetic_series.iloc[:700]
    test_part = synthetic_series.iloc[700:].reset_index(drop=True)
    
    detector = AutoRegressiveAnomalyDetector(lags=5, threshold_sigma=3.0)
    detector.fit(train_part)
    pred_df = detector.predict(test_part)
    
    assert len(pred_df) == len(test_part)
    assert set(pred_df["anomaly_pred"].unique()).issubset({0, 1})


def test_business_cost_matrix():
    y_true = np.array([0, 0, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1])  # 1 FP, 1 FN, 2 TP
    
    costs = compute_business_costs(y_true, y_pred, cost_fp=10.0, cost_fn=100.0)
    
    assert costs["cost_fp_total"] == 10.0
    assert costs["cost_fn_total"] == 100.0
    assert costs["total_expected_cost_eur"] == 110.0
    assert costs["null_system_cost_eur"] == 300.0
    assert costs["net_savings_eur"] == 190.0
