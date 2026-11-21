import numpy as np
def equation(t, A, p):
    return -(p[0]*A**p[1])*(1+p[2]*np.sin(p[3]*t+p[4]))
