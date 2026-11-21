def equation(t,P,params):
    return -params[0]*P**1.5*(1+params[1]*np.sin(params[2]*t+params[3]))+params[4]*P