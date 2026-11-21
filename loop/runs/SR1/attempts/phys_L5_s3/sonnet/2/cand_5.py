def equation(t,x,v,params):
    return params[0]+params[1]*np.sin(params[2]*x)+params[3]*t**params[4]+params[5]*v