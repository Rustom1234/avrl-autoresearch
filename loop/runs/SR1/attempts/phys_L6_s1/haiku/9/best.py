import numpy as np
def equation(t, x, v, params):
    w=params[9]
    C=np.cos(w*t); S=np.sin(w*t)
    return params[0]+params[1]*x+params[2]*v+(params[3]+params[5]*v+params[7]*t*v)*C+(params[4]+params[6]*v+params[8]*t*v)*S
