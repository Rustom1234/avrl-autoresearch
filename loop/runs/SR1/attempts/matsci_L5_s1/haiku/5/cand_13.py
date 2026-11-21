import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*T+p[2]*np.log(T)+p[3]/T+p[4]*np.sqrt(eps)
