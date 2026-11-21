import numpy as np
def equation(t, A, params):
    return -(params[0]*(1+params[1]*t)**params[4])*A/(1+params[2]*A+params[3]*A**2)