import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*np.log(eps) + params[2]*T + params[3]*np.sqrt(eps) + params[4]*eps + params[5]*T*np.log(eps) + params[6]*np.sqrt(eps)*T + params[7]*eps*T + params[8]*T**2 + params[9]*np.log(eps)**2
