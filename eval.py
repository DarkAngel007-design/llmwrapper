import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_multitask(y_true, y_pred, w):
    """
    y_true, y_pred, w: numpy arrays of shape (N, T)
    """

    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    w =w.reshape(w.shape[0], -1)

    roc_scores , pr_scores = [], []

    

    for t in range(y_true.shape[1]):
        mask = w[:, t] > 0
        if mask.sum() ==0:
            continue

        y_t = y_true[mask, t].astype(np.float32)
        y_p = y_pred[mask, t].astype(np.float32)

        if len(np.unique(y_t)) < 2 or np.std(y_p) ==0:
            continue
        roc_scores.append(
            roc_auc_score(y_t, y_p)
        )
        pr_scores.append(
            average_precision_score(y_t, y_p)
        )

    return{
        "roc_auc": float(np.mean(roc_scores)) if roc_scores else None,
        "pr_auc": float(np.mean(pr_scores)) if pr_scores else None,
        "n_valid_tasks": len(roc_scores),
    }
