import numpy as np
def equation(t, A, p):
    return p[0]*np.exp(-p[1]*t)*A**p[2] - p[3]*np.exp(-p[4]*t)*A**p[5]
