import numpy as np
def equation(eps, T, params):
    L=np.log(eps)
    return params[0]+params[1]*L+params[2]*L**2+params[3]*T+params[4]*T*L+params[5]*T*L**2+params[6]*eps+params[7]*T*eps+params[8]*T**2+params[9]*T**2*L
