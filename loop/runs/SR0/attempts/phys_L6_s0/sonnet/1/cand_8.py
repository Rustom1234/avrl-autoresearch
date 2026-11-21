def equation(t,x,v,params):
    return params[0]*np.sqrt(t)+params[1]*x*v+params[2]*(1-np.exp(-params[3]*x))+params[4]+params[5]*v