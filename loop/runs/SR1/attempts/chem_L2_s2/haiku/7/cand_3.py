import numpy as np
def equation(t, A, params):
    return (params[0]+params[1]*t)*A**2 + (params[2]+params[3]*t)*A + params[4] + params[5]*t