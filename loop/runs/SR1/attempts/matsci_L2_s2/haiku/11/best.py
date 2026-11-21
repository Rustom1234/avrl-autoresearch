import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*np.log(eps) + params[2]*eps + params[3]*np.sqrt(eps)
            + params[4]/np.sqrt(eps) + params[5]*T**2*eps**params[6] + params[7]*eps**2 + params[8]*T**2*eps**params[9])
