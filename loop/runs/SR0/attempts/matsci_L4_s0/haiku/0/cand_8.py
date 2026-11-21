import numpy as np
def equation(eps, T, params):
    x=np.maximum(eps-params[0],1e-9)
    return (params[1]+params[2]*T)*(1-np.exp(-(params[3]+params[4]*T)*x)) + params[5]*T + params[6]*eps + params[7]*T*eps + params[8]*T**2 + params[9]*eps**2
