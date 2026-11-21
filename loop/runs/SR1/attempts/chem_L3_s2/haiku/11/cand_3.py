import numpy as np
def equation(t, A, p):
    return (p[0] + p[1]*A**p[2])*(1-np.exp(-p[3]*t)) + p[4]*A**2*np.exp(-p[5]*t)
