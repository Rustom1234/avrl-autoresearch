import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*T + params[2]*T**2
            + (params[3] + params[4]*T)*np.exp(-params[5]*eps)
            + (params[6] + params[7]*T)*eps**2 + params[8]*eps + params[9]*np.log(eps))
