import numpy as np
def equation(t, A, p):
    return (p[0]+p[1]*t+p[2]*t**2+p[3]*t**3)*A + (p[4]+p[5]*t+p[6]*t**2)*A**2 + p[7]+p[8]*t+p[9]*t**2 + 0.0*A
