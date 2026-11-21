def equation(t,P,params):
    return params[0]*(1-P/params[1])+params[2]*P*np.exp(-params[3]*P)+params[4]*np.exp(-params[5]*P)