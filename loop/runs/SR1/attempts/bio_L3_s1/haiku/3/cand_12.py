import numpy as np
def equation(t, P, params):
    return (params[0]+params[1]*t)*P*(1-P/params[2]) + params[3] + params[4]*P + params[5]/P + params[6]*P**2
