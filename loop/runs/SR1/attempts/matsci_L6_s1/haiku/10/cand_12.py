import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*T + (p[2] + p[3]*T)*np.abs(eps)**p[4] + (p[5] + p[6]*T**2)*eps**p[7] + p[8]*np.log(eps)*T + p[9]*eps*T
