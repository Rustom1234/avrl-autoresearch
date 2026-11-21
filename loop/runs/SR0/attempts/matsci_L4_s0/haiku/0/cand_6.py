import numpy as np
def equation(eps, T, params):
    return (params[0]+params[1]*T)*np.log(params[2]+eps)+params[3]+params[4]*T+params[5]*eps+params[6]*T*eps+params[7]*T**2+params[8]*np.log(eps)
