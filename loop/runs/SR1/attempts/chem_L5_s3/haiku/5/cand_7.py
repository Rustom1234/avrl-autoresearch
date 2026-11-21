import numpy as np
def equation(t, A, p):
    return (p[0] + p[1]*t + p[2]*t**2 + p[3]*t**3) * A**p[4] + p[5]
