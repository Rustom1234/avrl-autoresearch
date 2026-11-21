def equation(t,P,params):
    return -params[0]*P*np.exp(params[1]*t)-params[2]*P**params[3]*(1+params[4]*t)