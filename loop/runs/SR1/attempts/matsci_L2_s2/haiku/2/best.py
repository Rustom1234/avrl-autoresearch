import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*np.exp(-p[2]*eps)+p[3]*eps+p[4]*np.log(eps)+p[5]*np.exp(p[6]*eps*T)
