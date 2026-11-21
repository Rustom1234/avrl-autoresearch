def equation(t,x,v,params):
    return params[0]+params[1]*x*np.exp(-params[2]*x)+params[3]*t**params[4]+params[5]*v