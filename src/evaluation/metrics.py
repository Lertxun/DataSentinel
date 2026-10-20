"""
Métricas de evaluación rigurosa para detección de anomalías en series temporales.
Implementa métricas por punto y métricas basadas en eventos/rangos para evitar
el sesgo del Point Adjustment indiscriminado.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def evaluate_pointwise(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> dict:
    """
    Evaluación punto a punto estándar (sin point adjustment).
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Falsa alarma / False Positive Rate
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "total_false_alarms": int(fp),
        "detected_positives": int(np.sum(y_pred))
    }
    
    if y_score is not None:
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["pr_auc"] = 0.0
            
    return metrics


def evaluate_event_aware(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluación por segmentos/eventos anómalos:
    Si una anomalía es un rango continuado de fallos, verifica si el modelo
    detectó al menos un punto dentro del evento (Event-Recall) y mide cuántas
    alertas fuera del evento fueron falsas alarmas.
    """
    # Identificar segmentos continuos de y_true == 1
    diff = np.diff(np.pad(y_true.astype(int), (1, 1), 'constant'))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    num_events = len(starts)
    if num_events == 0:
        return {"event_recall": 1.0, "num_events": 0}
        
    detected_events = 0
    first_detection_delays = []
    
    for s, e in zip(starts, ends):
        segment_pred = y_pred[s:e]
        if np.any(segment_pred == 1):
            detected_events += 1
            first_idx = np.where(segment_pred == 1)[0][0]
            first_detection_delays.append(first_idx)
            
    event_recall = detected_events / num_events
    avg_delay = float(np.mean(first_detection_delays)) if first_detection_delays else None
    
    return {
        "num_events": num_events,
        "detected_events": detected_events,
        "event_recall": float(event_recall),
        "avg_detection_delay_points": avg_delay
    }
