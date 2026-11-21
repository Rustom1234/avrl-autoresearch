import numpy as np
def equation(t, x, v, params):
    g = t**params[7]
    return (params[0]+params[1]*x+params[2]*x**2+params[3]*x**3+params[4]*v+params[5]*x*v
            +g*(params[6]+params[8]*x+params[9]*x**2))
