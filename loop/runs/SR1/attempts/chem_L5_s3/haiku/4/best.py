import numpy as np
def equation(t, A, params):
    return (params[0]+params[1]*t+params[2]*t**2+params[6]*t**3)*A/(params[3]+A) + (params[4]+params[5]*t)*A
