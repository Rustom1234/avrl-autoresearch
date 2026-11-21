def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t)) + params[2]*P*np.log(params[3]*P) + params[4]*P*t+params[5]*P**2*t