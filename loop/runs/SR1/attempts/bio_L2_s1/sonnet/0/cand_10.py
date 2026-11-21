def equation(t,P,params):
    return params[0]+params[1]*np.log(P)+params[2]*P*np.exp(-params[3]*t)+params[4]*t+params[5]*P