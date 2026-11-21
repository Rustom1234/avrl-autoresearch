import numpy as np
def equation(t, x, v, params):
    return (params[0]+params[1]*x+params[2]*x**2+params[3]*x**3+params[4]*x**4+params[5]*v+params[6]*v**2+params[7]*v**3+params[8]*t+params[9]*t**2)
