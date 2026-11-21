import numpy as np
def equation(eps, T, params):
    s = np.sqrt(eps)
    return (params[0] + params[1]*np.log(eps) + params[2]*s + params[3]*eps
            + T*(params[4]*s + params[5]*eps + params[6]*eps**2)
            + params[7]*T**2*eps**2 + params[8]*T**2*eps**3 + params[9]*T**3*eps**2)