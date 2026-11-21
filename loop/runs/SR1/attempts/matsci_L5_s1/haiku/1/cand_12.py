import numpy as np
def equation(eps, T, params):
    return params[0]+params[1]*eps+params[2]*T+params[3]*eps*eps+params[4]*eps*T+params[5]*T*T+(params[6]+params[7]*T)*np.sin(params[8]*eps*T**params[9]+params[7]*0)
