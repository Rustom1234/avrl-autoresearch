import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x/v + params[3]*x**params[4] + params[5]/(1+x**2)
