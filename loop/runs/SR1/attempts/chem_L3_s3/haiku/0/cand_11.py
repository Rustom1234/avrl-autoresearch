import numpy as np
def equation(t, A, p):
    return p[0] + p[1]*np.log(t) + p[2]/t + p[3]*A*t**p[4] + p[5]*A**2*t**p[6]
