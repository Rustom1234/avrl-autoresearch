import numpy as np
def equation(t, A, p):
    return (p[0]+p[1]*t+p[2]*t**2)*A + (p[3]+p[4]*t)*A**2 + p[5]+p[6]*t
