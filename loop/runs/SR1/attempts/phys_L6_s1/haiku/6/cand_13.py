import numpy as np
def equation(t, x, v, params):
    w=params[0]
    S=np.sin(w*t); C=np.cos(w*t)
    return (params[1] + params[2]*x + params[3]*v + (params[4]+params[5]*v)*S + (params[6]+params[7]*v)*C
            + params[8]*x*S + params[9]*x*C)