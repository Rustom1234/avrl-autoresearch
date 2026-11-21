import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*np.log(eps)/T + p[2]*eps*np.exp(-eps) + p[3]*np.exp(-eps) + p[4]*eps**2*T**2 + p[5]*np.log(eps)*T + p[6]*np.exp(-2*eps)*T**2 + p[7]*eps**(-1)*T**2 + p[8]*np.sqrt(eps)*T