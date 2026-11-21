import numpy as np
def equation(t, A, params):
    k = params[0]*(t**params[1])
    return -k*A**params[2]/(1+params[3]*A**params[4])
