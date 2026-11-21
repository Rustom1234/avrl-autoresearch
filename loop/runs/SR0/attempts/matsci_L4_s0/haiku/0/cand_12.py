import numpy as np
def equation(eps, T, params):
    L=np.log(eps)
    return params[0]+params[1]*L+params[2]*L**2+params[3]*T+params[4]*T*L+params[5]*T**2+params[6]*T**3+params[7]*eps+params[8]*T**2*L+params[9]*eps**2
