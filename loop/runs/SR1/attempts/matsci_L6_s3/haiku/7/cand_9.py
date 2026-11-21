import numpy as np
def equation(eps, T, params):
    p=params
    return -(p[0]+p[1]*T**p[2]) - (p[3]+p[4]*T)*np.abs(eps)**p[5] - p[6]*np.abs(eps)*np.exp(-p[7]*np.abs(eps))*T**p[8]
