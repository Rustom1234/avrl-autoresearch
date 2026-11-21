import numpy as np
def equation(eps, T, params):
    return params[0]/T + params[1]*eps**3/T + params[2]*T*np.log(T)