import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*T+(p[2]+p[3]*T)*eps**(p[4]+p[5]*T)+(p[6]+p[7]*T)/eps
