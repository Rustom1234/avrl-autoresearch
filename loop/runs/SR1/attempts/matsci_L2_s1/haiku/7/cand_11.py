import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*np.exp(-p[2]*T) + p[3]*T + (p[4] + p[5]*T)*eps
