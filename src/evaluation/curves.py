"""
Módulo de cálculo de curvas Precision-Recall (PR) y ROC.
Permite evaluar la capacidad de discriminación intrínseca de los anomaly scores
independientemente del umbral binario fijado de antemano.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


def compute_pr_curve(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """
    Calcula la curva Precision-Recall y el área bajo la curva (PR-AUC / Average Precision).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    
    return {
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "thresholds": thresholds.tolist(),
        "pr_auc": float(pr_auc)
    }


def compute_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """
    Calcula la curva ROC y el ROC-AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": float(roc_auc)
    }
