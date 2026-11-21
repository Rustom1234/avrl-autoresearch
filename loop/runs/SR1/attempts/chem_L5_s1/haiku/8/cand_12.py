import numpy as np
def equation(t, A, params):
    return -(params[0]+params[1]*t*A**params[4])*A**params[2]/(1+params[3]*A**2)
