import numpy as np
def equation(t, A, p):
    return p[0] + p[1]*A**p[2] + p[3]*t**p[4]/A**p[5] + p[6]*t**p[7]*A**p[8]
