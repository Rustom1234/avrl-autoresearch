def equation(t,P,params):
    return params[0]*P*(1-P/params[1]) - params[2]*(1-np.exp(-params[3]*t))+params[4]*P*np.exp(-params[5]*t)