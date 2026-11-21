def equation(t,P,params):
    return params[0]+params[1]*np.log(P)+params[2]*t+params[3]*P*np.exp(-params[4]*t)+params[5]*np.sin(params[6]*t+params[7])