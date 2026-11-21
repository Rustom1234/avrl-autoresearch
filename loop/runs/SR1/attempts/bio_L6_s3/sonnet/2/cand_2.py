def equation(t,P,params):
    return -params[0]*np.tanh(params[1]*t)*(1+params[4]*t)+params[2]*P*(1-P/params[3])+params[5]*P**2*np.exp(-params[6]*t)