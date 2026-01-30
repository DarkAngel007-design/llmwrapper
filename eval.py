import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_multitask(y_true, y_pred, w):
    """
    y_true, y_pred, w: numpy arrays of shape (N, T)
    """

    roc_scores , pr_scores = [], []

    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    w = w.reshape(w.shape[0], -1)

    for t in range(y_true.shape[1]):
        mask = w[:, t]==1
        if mask.sum() ==0:
            continue

        if len(np.unique(y_t)) < 2:
            continue
            
        roc_scores.append(
            roc_auc_score(y_true[mask, t], y_pred[mask, t])
        )
        pr_scores.append(
            average_precision_score(y_true[mask, t], y_pred[mask, t])
        )

    return{
        "roc_auc": float(np.mean(roc_scores)) if roc_scores else None,
        "pr_auc": float(np.mean(pr_scores)) if pr_scores else None,
        "n_valid_tasks": len(roc_scores)
    }

