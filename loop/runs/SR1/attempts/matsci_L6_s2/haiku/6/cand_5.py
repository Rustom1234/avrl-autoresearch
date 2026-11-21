import numpy as np
def equation(eps, T, params):
    return params[0]*np.exp(params[1]*T)*eps**(params[2]+params[3]*T) + params[4]*np.exp(params[5]*T)*eps**params[6]
