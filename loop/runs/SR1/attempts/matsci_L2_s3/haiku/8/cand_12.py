import numpy as np
def equation(eps, T, params):
    p = params
    s = np.sqrt(T)
    return (p[0] + p[1]*T + p[2]*T**2 + p[3]*s) + eps*(p[4] + p[5]*T + p[6]*s) + eps**2*(p[7] + p[8]*s + p[9]*T)