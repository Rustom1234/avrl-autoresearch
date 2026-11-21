import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*T + (params[2] + params[3]*T)*eps*np.log(eps)
            + (params[4] + params[5]*T)*eps + (params[6] + params[7]*T)*eps**2 + params[8]*T**2)
