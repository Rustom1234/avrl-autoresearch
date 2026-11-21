import numpy as np
def equation(eps, T, params):
    return -(params[0]+params[1]*eps**params[2])*np.exp(params[3]*T)*(1+params[4]*T)
