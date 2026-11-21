import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*T + (p[2]+p[3]*T)*np.exp(-(p[4]+p[5]*T)*eps) + (p[6]+p[7]*T)*eps
