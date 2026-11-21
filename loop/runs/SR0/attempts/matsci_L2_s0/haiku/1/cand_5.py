import numpy as np
def equation(eps, T, params):
    return T*(params[0] + params[1]*(1-np.exp(-params[2]*eps)) + params[3]*eps) * (1 + params[4]*T) 
