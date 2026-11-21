import numpy as np
def equation(t, P, params):
    w=params[0]
    s=np.sin(w*t); c=np.cos(w*t)
    return (params[1]+params[2]*P+params[3]*P**2)*1.0 + (params[4]*P+params[5]*P**2)*s + (params[6]*P+params[7]*P**2)*c + params[8]*s + params[9]*c
