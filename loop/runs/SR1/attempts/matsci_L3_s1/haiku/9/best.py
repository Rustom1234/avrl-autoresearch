import numpy as np
def equation(eps, T, params):
    p = params
    return (p[0] + p[1]*T**2) + (p[2] + p[3]*np.log(T) + p[5]*0) * eps*np.exp(-p[4]*eps)
