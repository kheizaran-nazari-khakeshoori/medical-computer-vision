"""Examining bias in dataset."""
from collections import Counter
def check_label_balance(labels):
    c=Counter(labels)
    total=len(labels)
    return {k: v/total for k,v in c.items()}
def check_age_bias(metadata):
    # stub
    return {}
