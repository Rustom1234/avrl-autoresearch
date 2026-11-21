import numpy as np
def equation(t, A, params):
    p = params
    return p[0] + p[1]*t**p[4] + p[2]/A + p[3]*A/t
