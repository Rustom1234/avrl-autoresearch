import numpy as np
def equation(eps, T, params):
    p=params
    e=np.abs(eps)
    return -(p[0]+p[1]*np.log(T)) + p[2]*T**p[3]*e**p[4] - p[5]*e + p[6]*np.log(T)*e
