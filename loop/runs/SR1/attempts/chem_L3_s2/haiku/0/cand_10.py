import numpy as np
def equation(t, A, params):
    return params[0]*t/(params[1]+t) + A**params[2]*(params[3]/(1+params[4]*t)) + params[5]*A**params[2]*t/(params[1]+t)
