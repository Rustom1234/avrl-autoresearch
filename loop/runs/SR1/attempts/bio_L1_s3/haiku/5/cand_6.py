import numpy as np
def equation(t, P, params):
    return params[0]*P*np.log(params[1]/P)*(P>0) + params[2]*P*(1-P/params[3])
