import numpy as np
def equation(t, A, params):
    ramp = 1 - np.exp(-params[0]*t)
    return ramp*(params[1] + params[2]*t + (params[3] + params[4]*t)*A + (params[5]+params[6]*t)*A**2)