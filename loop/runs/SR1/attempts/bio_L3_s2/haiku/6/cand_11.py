import numpy as np
def equation(t, P, params):
    w=params[9]
    s=np.sin(w*t); c=np.cos(w*t)
    return (params[0]+params[1]*t+P*(params[2]+params[3]*s+params[4]*c)+params[5]*P**2+params[6]*P**3
            +params[7]*s+params[8]*c)
