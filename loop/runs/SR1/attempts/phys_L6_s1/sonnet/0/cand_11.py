def equation(t,x,v,params):
    return params[0]+params[1]*np.sin(params[2]*t+params[3])+params[4]*v*np.exp(params[5]*t)+params[6]*v