import numpy as np
def equation(t,P,params):
    return params[0]+params[1]*np.exp(params[2]*t)+params[3]*np.log(P)+params[4]*t*P+params[5]/P+params[6]*P+params[7]*P**2+params[8]*t
