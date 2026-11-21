import numpy as np
def equation(eps, T, params):
    w=params[0]
    return (params[1]+params[2]*T+params[3]*T*T)*np.sin(w*eps)+(params[4]+params[5]*T+params[6]*T*T)*np.cos(w*eps)+(params[7]+params[8]*T)*eps+params[9]
