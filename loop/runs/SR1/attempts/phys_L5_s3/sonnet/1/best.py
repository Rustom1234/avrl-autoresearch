def equation(t,x,v,params):
    return params[0]*x*np.exp(-params[1]*x)+params[2]*t**params[3]+params[4]+params[5]*v