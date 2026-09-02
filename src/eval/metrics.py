"""
Evaluation metrics matching paper Section IV-B exactly (Eq 6-12):
  ACC, macro-averaged precision/recall/F1, Cohen's kappa.
Also reports per-class precision/recall/F1/specificity (paper Table IV format).

Uses sklearn under the hood but computes kappa via the paper's own Eq 11-12
formula explicitly (rather than trusting sklearn's cohen_kappa_score
blindly) so the arithmetic is auditable against the paper text.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(CLASS_NAMES)

    acc = (y_true == y_pred).mean()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    # Eq 10: Macro-F1 = harmonic mean of macro-P and macro-R (paper's specific
    # definition -- NOT the same as sklearn's per-class-then-average F1 in
    # general, though they're close in balanced cases. Compute per paper's Eq.)
    macro_f1 = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0 else 0.0
    )

    # Eq 11-12: Cohen's kappa, explicit paper formula
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    N = cm.sum()
    po = np.trace(cm) / N
    row_sums = cm.sum(axis=1)   # n_k1: true-label counts per class
    col_sums = cm.sum(axis=0)   # n_k2: predicted-label counts per class
    pe = (row_sums * col_sums).sum() / (N ** 2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    # Per-class: precision, recall, F1, specificity (Table IV format)
    per_class = {}
    for k, name in enumerate(CLASS_NAMES):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = N - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class[name] = {
            "precision": float(precision[k]),
            "recall": float(recall[k]),
            "f1": float(f1[k]),
            "specificity": float(spec),
            "support": int(support[k]),
        }

    return {
        "acc": float(acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_metrics(metrics: dict) -> str:
    lines = [
        f"ACC={metrics['acc']*100:.1f}%  MF1={metrics['macro_f1']*100:.1f}%  "
        f"Kappa={metrics['kappa']:.3f}",
        "Per-class recall (acc): " + " ".join(
            f"{name}={metrics['per_class'][name]['recall']*100:.1f}%" for name in CLASS_NAMES
        ),
    ]
    return "\n".join(lines)