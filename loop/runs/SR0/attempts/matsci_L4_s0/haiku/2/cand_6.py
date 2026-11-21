import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*T + params[2]*T**2)*(1-np.exp(-(eps-params[3])/params[4])) + params[5]*T*eps
