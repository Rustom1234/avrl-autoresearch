import numpy as np
def equation(eps, T, params):
    L=np.log(eps)
    return params[0]+params[1]*L+params[2]*L**2+params[3]*eps+params[4]*eps**2+params[5]*T+params[6]*T*L+params[7]*T*L**2+params[8]*T*eps+params[9]*T**2
