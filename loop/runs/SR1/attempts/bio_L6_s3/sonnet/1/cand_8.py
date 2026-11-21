def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t)) + params[2]*P + params[3]*P**params[4]/(1+np.exp(-params[5]*(t-params[6])))