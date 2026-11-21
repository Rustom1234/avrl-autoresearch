import numpy as np
def equation(t, A, params):
    return -(params[0] + params[1]*np.sqrt(t) + params[2]*np.exp(-params[3]*t)*(1+params[5]*A**2)) - params[4]*np.sqrt(t)*A**2
