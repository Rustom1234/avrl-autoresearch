import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*T**0.6 + params[2]*T**2 + params[3]*eps**1.5