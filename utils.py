import numpy as np


def extract_min(structure, function):
    min_el = None
    min_score = np.inf
    for el in structure:
        score = function(el)
        if score < min_score:
            min_score = score
            min_el = el
    return min_el
