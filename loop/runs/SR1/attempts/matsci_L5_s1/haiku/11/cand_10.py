import numpy as np
def equation(eps, T, params):
    p=params
    return (p[0]+p[1]*T+p[2]*T**2)*np.exp(-p[3]*eps)+p[4]+p[5]*T+p[6]*eps+p[7]*eps*T