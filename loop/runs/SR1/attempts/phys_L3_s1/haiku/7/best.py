import numpy as np
def equation(t, x, v, params):
    w=params[1]
    s=np.sin(w*t); c=np.cos(w*t); sv=np.sqrt(v)
    return params[0]*s+params[2]+params[3]*t+params[4]*v+params[5]*sv+params[6]*s*sv+params[7]*c+params[8]*s*t+params[9]*c*t
