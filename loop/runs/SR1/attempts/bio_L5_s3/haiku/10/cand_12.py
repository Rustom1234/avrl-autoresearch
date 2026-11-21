import numpy as np
def equation(t, P, params):
    return params[0]*P*(1-P/params[1]) - params[2]*P**2/(params[3]**2+P**2) - params[4]*t + params[5]