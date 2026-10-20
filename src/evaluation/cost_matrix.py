"""
Módulo de Evaluación de Costes Económicos y Operativos Asimétricos.
Calcula el Coste Total Esperado (Total Expected Cost - TEC) y el ahorro
económico relativo en función del impacto real de Falsos Positivos vs Falsos Negativos.
"""
import numpy as np


def compute_business_costs(y_true: np.ndarray, y_pred: np.ndarray, cost_fp: float = 15.0, cost_fn: float = 2500.0) -> dict:
    """
    Parámetros:
    - cost_fp: coste económico de investigar una falsa alarma (horas de analista/ingeniero).
    - cost_fn: coste económico de omitir una anomalía crítica real (rotura de stock, parada industrial, fuga de seguridad).
    """
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    cost_from_fp = fp * cost_fp
    cost_from_fn = fn * cost_fn
    total_cost = cost_from_fp + cost_from_fn

    # Coste del escenario nulo (no tener ningún sistema y asumir todas las anomalías sin detectar)
    total_anomalies = int(np.sum(y_true == 1))
    null_system_cost = total_anomalies * cost_fn

    # Ahorro neto económico del modelo respecto a no tener detección
    net_savings = null_system_cost - total_cost
    savings_pct = (net_savings / null_system_cost * 100.0) if null_system_cost > 0 else 0.0

    return {
        "cost_fp_total": cost_from_fp,
        "cost_fn_total": cost_from_fn,
        "total_expected_cost_eur": total_cost,
        "null_system_cost_eur": null_system_cost,
        "net_savings_eur": net_savings,
        "savings_percentage": savings_pct
    }
