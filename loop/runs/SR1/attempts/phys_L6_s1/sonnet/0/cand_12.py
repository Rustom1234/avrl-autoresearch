def equation(t,x,v,params):
    return params[0]+params[1]*t+params[2]*t**2+params[3]*np.sin(params[4]*t)+params[5]*v*np.exp(params[6]*t)+params[7]*v