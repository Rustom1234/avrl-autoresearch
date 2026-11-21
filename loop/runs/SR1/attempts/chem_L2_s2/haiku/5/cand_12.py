import numpy as np
def equation(t, A, p):
    return p[0] + p[1]*A + p[2]*t**p[3]/A + p[4]*t**p[5] + p[6]*A*t**p[7]
