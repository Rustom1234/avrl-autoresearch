import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*T**2 + (params[2] + params[3]*T)*(1-np.exp(-(params[4]+params[5]*T)*eps)) + params[6]*T
