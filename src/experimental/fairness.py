"""Needing review of model fairness."""
from sklearn.metrics import accuracy_score
def per_group_accuracy(y_true, y_pred, groups):
    result={}
    for g in set(groups):
        idx=[i for i, gg in enumerate(groups) if gg==g]
        yt=[y_true[i] for i in idx]
        yp=[y_pred[i] for i in idx]
        result[g]= accuracy_score(yt, yp) if yt else 0
    return result
