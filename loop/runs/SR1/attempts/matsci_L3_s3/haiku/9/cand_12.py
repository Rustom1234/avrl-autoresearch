import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*T)*np.exp(params[2]*T)*eps**(params[3]+params[4]*T) + params[5] + params[6]*T + params[7]*T*eps
