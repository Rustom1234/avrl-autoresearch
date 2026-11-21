def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t))+params[2]*P**params[3]*(t-params[4])+params[5]*P+params[6]*np.exp(-params[7]*t)*P**2