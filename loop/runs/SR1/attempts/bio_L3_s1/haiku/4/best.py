import numpy as np
def equation(t, P, params):
    p = params
    return (p[0] + p[1]*t) + (p[2] + p[3]*t)*P + (p[4] + p[5]*t)*P**2 + p[6]/P + p[7]*t/P
