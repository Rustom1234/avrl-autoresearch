import numpy as np
def equation(t, A, params):
    return (params[0] + params[1]*t + params[2]*t**2)*A**(-0.75) + (params[3] + params[4]*t + params[5]*t**2)*A**(-0.25)
