import numpy as np
def equation(t, A, p):
    return p[0] + p[1]/np.sqrt(t) + p[2]*A*t + p[3]*A**2 + p[4]*A**3 + p[5]*A + p[6]*t
