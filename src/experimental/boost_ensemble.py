"""Boosting model accuracy with ensemble voting."""



def majority_vote(preds):
    import collections

    return collections.Counter(preds).most_common(1)[0][0]


def weighted_ensemble(probs_list, weights=None):

    weights = weights or [1 / len(probs_list)] * len(probs_list)
    return sum(w * p for w, p in zip(weights, probs_list))
