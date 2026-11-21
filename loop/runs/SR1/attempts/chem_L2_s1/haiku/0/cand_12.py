import numpy as np
def equation(t, A, params):
    p=params
    return -((p[0]+p[1]*t**p[2])*A/(1+(p[3]+p[4]*t**p[5])*A))