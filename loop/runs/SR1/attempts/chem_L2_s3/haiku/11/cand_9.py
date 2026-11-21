import numpy as np
def equation(t, A, params):
    k=params[0]*A**params[1]
    q=params[2]+params[3]*t+params[4]/t+params[5]*np.sqrt(t)
    return -k + (params[6]+params[7]*A+params[8]*A**2)*q + params[9]*A
