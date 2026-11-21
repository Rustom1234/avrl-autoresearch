import numpy as np
def equation(eps, T, params):
    L=np.log(eps)
    return (params[0]+params[1]*T+params[2]*T**2)*L + (params[3]+params[4]*T)*eps**params[5] + params[6]+params[7]*T+params[8]*T**2 + params[9]*T**3
