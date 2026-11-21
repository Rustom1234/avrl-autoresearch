import numpy as np
def equation(eps, T, params):
    return params[0]+params[1]*T+params[2]*T**2+params[3]*T**3+(params[4]+params[5]*T)*eps+(params[6]+params[7]*T)*np.sqrt(eps)+params[8]*eps**2+params[9]*eps**2*T
