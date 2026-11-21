def equation(t,x,v,params):
    return params[0]+params[1]*np.sin(params[2]*t)+params[3]*v*np.sin(params[4]*t)+params[5]*x+params[6]*v+params[7]*np.cos(params[2]*t)