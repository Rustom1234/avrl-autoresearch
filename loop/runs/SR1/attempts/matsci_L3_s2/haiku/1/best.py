import numpy as np
def equation(eps, T, params):
    L=np.log(T); I=1/T; E=np.exp(-eps/params[9])
    return (params[0] + params[1]*L + params[2]*I + (params[3] + params[4]*L + params[5]*I)*E
            + params[6]*eps + params[7]*eps**2 + params[8]*eps*L)
