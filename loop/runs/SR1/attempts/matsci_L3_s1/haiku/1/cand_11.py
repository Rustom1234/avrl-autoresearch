import numpy as np
def equation(eps, T, params):
    a = params[0] + params[1]*T
    c = params[2] + params[3]*T
    b = params[4] + params[5]*T
    return (a + c*(1-np.exp(-b*eps)))*np.exp(-params[6]*eps**params[7])
