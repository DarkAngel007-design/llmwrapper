import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_multitask(y_true, y_pred, w):
    """
    y_true, y_pred, w: numpy arrays of shape (N, T)
    """

    roc_scores , pr_scores = [], []

    for t in range(y_true.shape[1]):
        mask = w[:, t]==1
        if mask.sum() ==0:
            continue

        roc_scores.append(
            roc_auc_score(y_true[mask, t], y_pred[mask, t])
        )
        pr_scores.append(
            average_precision_score(y_true[mask, t], y_pred[mask, t])
        )

    return{
        "roc_auc": float(np.mean(roc_scores)),
        "pr_auc": float(np.mean(pr_scores))
    }