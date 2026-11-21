import numpy as np
def equation(t, A, params):
    return (A - params[0])*params[1]*A/(params[2]+A)*np.exp(-params[3]*t)*(1+params[4]*t)
