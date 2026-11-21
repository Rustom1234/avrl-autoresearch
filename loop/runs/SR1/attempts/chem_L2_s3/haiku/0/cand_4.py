import numpy as np
def equation(t, A, p):
    return -(p[0]*A**p[1])*(1+p[2]*t)
