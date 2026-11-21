import numpy as np
def equation(t, A, params):
    return -params[0]*(t+params[3])**params[1]*A**(params[2]+params[4]*t)
