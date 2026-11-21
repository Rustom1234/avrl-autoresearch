import numpy as np
def equation(t, P, params):
    return params[0]*P**params[2]*(1-P/params[1])
