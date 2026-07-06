"""Feature importance plot for model insights."""

import matplotlib.pyplot as plt


def plot_feature_importance(importance: dict, save_path=None):
    names = list(importance.keys())
    values = list(importance.values())
    plt.figure()
    plt.barh(names, values)
    plt.title("Feature Importance")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()
