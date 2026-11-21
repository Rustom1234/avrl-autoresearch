import numpy as np
def equation(t, A, p):
    return (p[0]*A**p[1] + p[2]) * (1 - np.exp(-p[3]*t)) * np.exp(-p[4]*t) + p[5]
