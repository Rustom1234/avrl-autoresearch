import numpy as np
def equation(t, A, p):
    return (p[0] + p[1]*A/(p[2]+A)) * (1 + p[3]*t + p[4]*t**2)
