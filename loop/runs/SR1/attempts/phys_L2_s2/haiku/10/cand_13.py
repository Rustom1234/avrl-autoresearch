import numpy as np
def equation(t, x, v, params):
    p=params
    return -p[0] - p[1]*x - p[2]*x**2 - p[3]*x**3 - p[4]/x - p[5]*x/v
