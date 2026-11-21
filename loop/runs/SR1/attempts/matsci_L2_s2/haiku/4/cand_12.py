import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*np.log(eps) + params[2]*eps**params[3]*T**params[4]
            + params[5]*eps*T + params[6]*T**params[7]*np.log(eps) + params[8]*T)