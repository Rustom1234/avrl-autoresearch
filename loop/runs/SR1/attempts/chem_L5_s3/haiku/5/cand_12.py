import numpy as np
def equation(t, A, p):
    return p[0]*t*np.exp(-p[1]*t) + p[2]*np.sqrt(A)
