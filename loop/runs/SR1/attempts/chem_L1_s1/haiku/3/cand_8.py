import numpy as np
def equation(t, A, params):
    return params[0]/A+params[1]+params[2]*A+params[3]*A**2+(params[4]+params[5]*A)*np.sin(params[6]*t+params[7])+params[8]*t
