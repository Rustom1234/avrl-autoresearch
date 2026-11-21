def equation(t,P,params):
    return params[0]*(1-np.exp(-params[1]*t))+params[2]*P+params[3]+params[4]*P*t+params[5]*P**2