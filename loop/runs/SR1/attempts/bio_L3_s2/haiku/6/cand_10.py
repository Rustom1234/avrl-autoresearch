import numpy as np
def equation(t, P, params):
    w=params[9]
    return (params[0]+params[1]*P+params[2]*P**2+params[3]*t+params[4]*np.sin(w*t)+params[5]*np.cos(w*t)
            +params[6]*P*np.sin(w*t)+params[7]*P*np.cos(w*t)+params[8]*P*t)
