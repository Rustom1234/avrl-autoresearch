import numpy as np
def equation(t, A, params):
    S = params[0]*t**params[1]/(params[2]**params[1]+t**params[1])/(1+params[3]*t**params[7])
    return S - params[4]*A**params[5]*t**params[6]
