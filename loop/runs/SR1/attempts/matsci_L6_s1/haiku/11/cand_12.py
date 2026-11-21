import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*T + p[2]*T**2 + (p[3] + p[4]*T)*eps**p[5] + p[6]*eps*T**2 + p[7]/T + p[8]*np.sin(eps)
