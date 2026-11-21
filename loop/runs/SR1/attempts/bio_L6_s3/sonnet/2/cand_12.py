def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t))+params[2]*np.exp(params[3]*P)*np.sin(params[4]*t+params[5])+params[6]*P