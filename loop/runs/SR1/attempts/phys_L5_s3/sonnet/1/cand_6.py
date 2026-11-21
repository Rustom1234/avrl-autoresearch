def equation(t,x,v,params):
    return params[0]*x*np.exp(-params[1]*x)+params[2]*np.cos(params[3]*t+params[4])+params[5]*v+params[6]