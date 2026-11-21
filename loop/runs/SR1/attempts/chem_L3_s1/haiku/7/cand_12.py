import numpy as np
def equation(t, A, params):
    return params[0]*(1+params[1]*t)**(-params[2])*(1-(A/params[3])**params[4])
