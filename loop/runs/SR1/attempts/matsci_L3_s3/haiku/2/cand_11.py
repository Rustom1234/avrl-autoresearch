import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*T+p[2]*T**2+(p[3]+p[4]*T+p[5]*T**2)*(1-np.exp(-p[6]*eps))