import numpy as np
def equation(eps, T, params):
    L=np.log(eps)
    return params[0]+params[1]*T+params[2]*T**2+(params[3]+params[4]*T)*L+params[5]*L**2+params[6]*eps+params[7]*T*eps
