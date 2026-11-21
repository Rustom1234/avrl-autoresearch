import numpy as np
def equation(eps, T, params):
    return params[0]*((eps+params[1])**0.375-params[1]**0.375)