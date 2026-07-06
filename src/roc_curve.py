"""ROC curve visualization for diagnostics."""

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


def plot_roc_curve(y_true, y_scores, save_path: str | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()
