import numpy as np
def equation(t, P, params):
    w=params[0]
    s=np.sin(w*t); c=np.cos(w*t)
    return params[1]*P + params[2]*P**2 + params[3] + (params[4]*P)*s + (params[5]*P)*c + params[6]*s + params[7]*c
