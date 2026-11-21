def equation(t,x,v,params):
    return params[0]+params[1]*v*np.exp(params[2]*x)+params[3]*x+params[4]*np.log(x)+params[5]*np.cos(params[6]*t+params[7])