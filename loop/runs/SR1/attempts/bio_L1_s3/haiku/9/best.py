import numpy as np
def equation(t, P, params):
    return params[0]*P**0.5*(1-(P/params[1])**1.5)