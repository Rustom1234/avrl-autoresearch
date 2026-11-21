import numpy as np
def equation(t, A, params):
    return params[0]+params[1]*A+params[2]*A**2+params[3]*A**3+(params[4]+params[5]*A)*np.sin(params[6]*t+params[7])+params[8]*t
