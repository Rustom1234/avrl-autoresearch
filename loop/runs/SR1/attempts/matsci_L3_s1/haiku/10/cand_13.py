import numpy as np
def equation(eps, T, params):
    E = params[0] + params[1]*np.sqrt(eps) + params[2]*eps + params[3]*eps**2 + params[4]*np.log(eps)
    return E*(1 + params[5]*T + params[6]*T**2 + params[7]/T + params[8]/T**2) + params[9]*T
