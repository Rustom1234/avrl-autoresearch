import numpy as np
def equation(t, P, params):
    r=params[0]; K0=params[1]; c=params[2]; H=params[3]; a=params[4]; n=params[5]
    return r*P*np.log(K0*np.exp(-c*t)/P) - H*t**n/(a**n+t**n) + params[6]*P*np.exp(-params[7]*t) + params[8]*P**2
