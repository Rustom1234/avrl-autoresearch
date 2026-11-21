import numpy as np
def equation(t,P,params):
    return params[0]+params[1]*t**2+params[2]*np.log(P)+params[3]*t*P+params[4]/P+params[5]*t+params[6]*P+params[7]*P**2
