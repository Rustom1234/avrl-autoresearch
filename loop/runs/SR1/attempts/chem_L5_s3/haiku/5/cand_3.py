import numpy as np
def equation(t, A, p):
    return (p[0] + p[1]*A + p[2]*A**2) * (1 + p[3]*t*np.exp(-p[4]*t)) + p[5]*t*np.exp(-p[6]*t)
