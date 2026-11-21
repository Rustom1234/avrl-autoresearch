import numpy as np
def equation(t, A, p):
    return p[0] + p[1]*np.log(A) + p[2]*np.log(t) + p[3]*A + p[4]/t
