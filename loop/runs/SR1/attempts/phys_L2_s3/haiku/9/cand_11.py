import numpy as np
def equation(t, x, v, params):
    g = np.sqrt(t)
    return (params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*v + params[5]*v**2 + params[6]*x*v
            + (params[7] + params[8]*x + params[9]*x**2)*g)