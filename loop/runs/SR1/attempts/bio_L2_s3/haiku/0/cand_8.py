import numpy as np
def equation(t,P,params):
    return (params[0]+params[1]*t)*np.exp(-params[2]*P)+params[3]+params[4]*P+params[5]*P**2+params[6]*t*P
