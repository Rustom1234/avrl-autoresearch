import numpy as np
def equation(t, P, params):
    return (params[0]*np.sqrt(P) + params[1]/P + params[2]*P**2 + params[3]*t**2
            + params[4]*t*P**2 + params[5]*t*P + params[6]*P**3 + params[7])
