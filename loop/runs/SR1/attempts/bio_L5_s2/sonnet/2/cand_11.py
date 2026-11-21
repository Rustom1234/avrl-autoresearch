def equation(t,P,params):
    return params[0]*(1-(P/params[1])**3)+params[3]*P*np.exp(-params[4]*P)