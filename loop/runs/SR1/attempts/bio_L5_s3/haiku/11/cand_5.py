import numpy as np
def equation(t, P, params):
    w=params[0]
    return params[1]*P+params[2]*P**2+params[3]*P**3+P*(params[4]*np.sin(w*t)+params[5]*np.cos(w*t))+P**2*(params[6]*np.sin(w*t)+params[7]*np.cos(w*t))+params[8]
