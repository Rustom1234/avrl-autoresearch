import numpy as np
def equation(t, A, params):
    p = params
    return (p[0] + p[1]*t + p[2]*t**2 + p[9]*t**3
            + (p[3] + p[4]*t + p[5]*t**2)*A
            + (p[6] + p[7]*t + p[8]*t**2)*A**2)
