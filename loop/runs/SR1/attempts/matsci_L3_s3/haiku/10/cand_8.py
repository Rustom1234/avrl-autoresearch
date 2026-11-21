import numpy as np
def equation(eps, T, params):
    return params[0]*eps**(params[1]+params[2]*T)*np.exp(params[3]*T) + params[4]*np.log(eps)+params[5]*T*np.log(eps)+params[6]
