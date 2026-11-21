import numpy as np
def equation(eps, T, params):
    g = params[1]*(1-np.exp(-params[2]*T**params[3])) 
    return -(params[0] + g) * (1 - params[4]*eps**params[5]) - params[6]*eps
