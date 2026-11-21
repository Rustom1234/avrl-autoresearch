import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*np.exp(-params[2]*T) + params[3]*np.sqrt(T)*np.exp(-params[4]*eps)
            + params[5]*eps**3/T + params[6]*np.exp(-params[7]*eps)/T + params[8]*T**3*eps**2 + params[9]*T)
