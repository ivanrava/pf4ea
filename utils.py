import numpy as np


def extract_min(structure: set, function):
    min_el = next(iter(structure))
    min_score = function(min_el)
    for el in structure:
        score = function(el)
        if score < min_score:
            min_score = score
            min_el = el
    return min_el
