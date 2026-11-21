import numpy as np
def equation(eps, T, params):
    return params[0]*(T+params[1])**params[2]*(1+params[3]*eps**2)**params[4]
