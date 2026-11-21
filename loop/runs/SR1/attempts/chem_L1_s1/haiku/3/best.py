import numpy as np
def equation(t, A, params):
    return params[0]/A + params[1]*np.exp(params[2]*A) + params[3]*A + params[4]*A**2 + params[5]*np.log(A) + params[6]*A*np.log(A)
