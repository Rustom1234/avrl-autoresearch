import numpy as np
def equation(t,P,params):
    return params[0]+params[1]*t+(params[2]+params[3]*t)*P+params[4]*P**2+params[5]*np.sin(params[6]*P)+params[7]*np.cos(params[6]*P)+params[8]*t**2
