import numpy as np
def equation(t, A, params):
    Vmax = params[0] + params[1]*t
    return -Vmax * A / (params[2] + A + A**2/params[3])
