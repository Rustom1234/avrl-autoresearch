import numpy as np
def equation(t, A, params):
    e = np.exp(-params[1]*t)
    return params[0]*e + params[2] + params[3]*A**2 + params[4]*A**2*t + params[5]*A*e + params[6]*A**2*e + params[7]*t
