import numpy as np
def equation(t, P, params):
    r=params[0]; K0=params[1]; c=params[2]; H=params[3]; k=params[4]
    return r*P*np.log(K0*np.exp(-c*t)/P) - H*(1-np.exp(-k*t))
