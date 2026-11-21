import numpy as np
def equation(t, A, p):
    return p[0]*t**p[1] + p[2]*A**p[3]*t**p[4] + p[5]
