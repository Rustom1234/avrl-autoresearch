import numpy as np
def equation(t, A, p):
    return -(p[0]+p[1]*t**p[2])*A**p[3]
