import numpy as np
def equation(eps, T, params):
    p=params
    e=np.abs(eps)
    return -(p[0]+p[1]*np.log(T)) - (p[2]+p[3]*np.log(T))*e**p[4] - p[5]*e*np.exp(-p[6]*e)*T**p[7]
