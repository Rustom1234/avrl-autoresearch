import numpy as np
def equation(t, x, v, params):
    w=params[0]
    C=np.cos(w*t); S=np.sin(w*t)
    return (params[1] + params[2]*C + params[3]*S + params[4]*x + params[5]*v
            + params[6]*x*x + params[7]*v*v + params[8]*x*v
            + params[9]*C*v)
