import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*eps**1.5 + params[2]*T**3
