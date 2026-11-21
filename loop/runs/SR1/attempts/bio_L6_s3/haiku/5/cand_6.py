import numpy as np
def equation(t, P, params):
    return -params[0]*np.tanh(params[1]*t)*(params[2]+params[3]*P+params[4]*P**2) + params[5]*P*(1-P/params[6])