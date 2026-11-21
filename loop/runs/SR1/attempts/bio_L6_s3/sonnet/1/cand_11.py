def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t)) + params[2]*P*(1-P/params[3]) + params[4]*P*np.sin(params[5]*t+params[6])