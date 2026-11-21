import numpy as np
def equation(t, P, params):
    return params[0]*P*(1-P/params[1]) + params[2]*np.sin(params[3]*t+params[4])*P + params[5]*np.sin(params[3]*t+params[4])
