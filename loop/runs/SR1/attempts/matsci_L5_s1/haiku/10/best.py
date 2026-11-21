import numpy as np
def equation(eps, T, params):
    L=np.log(eps); s=np.sqrt(T)
    return (params[0]+params[1]*s+params[2]*T+params[3]*T**2+(params[4]+params[5]*s)*L
            +params[6]*L**2+params[7]*eps+params[8]*eps**2+params[9]*L*eps)
