import numpy as np
def equation(eps, T, params):
    p=params
    e=np.abs(eps)
    return -(p[0]+p[1]*T**p[2]) - (p[3]+p[4]*T**p[5])*e**p[6] - p[7]*e*np.exp(-p[8]*e)*T**p[9]
