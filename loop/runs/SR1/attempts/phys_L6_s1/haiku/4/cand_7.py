import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return (params[1] + params[2]*v + params[3]*x + params[4]*v**2 + params[5]*x**3
            + (params[6] + params[7]*v)*np.cos(w*t) + (params[8] + params[9]*v)*np.sin(w*t))