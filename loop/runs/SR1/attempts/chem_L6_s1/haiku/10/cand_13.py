import numpy as np
def equation(t, A, params):
    S = A**2/(1+params[0]*A**2)
    return S*(params[1]+params[2]*t+params[3]*t**2) + A*(params[4]+params[5]*t+params[6]*t**2) + params[7]*A**2*t + params[8]*A**3*t**2 + params[9]*t**3