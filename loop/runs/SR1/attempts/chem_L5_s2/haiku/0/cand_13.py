import numpy as np
def equation(t, A, params):
    s = np.sqrt(t)
    return params[0] + params[1]*s + params[2]*t + params[3]*t**1.5 + params[4]*A*s + params[5]*A*t + params[6]*A**2*s + params[7]*A
