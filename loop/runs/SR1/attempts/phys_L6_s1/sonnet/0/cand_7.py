def equation(t,x,v,params):
    return params[0]+params[1]*np.sin(params[2]*t+params[3])+params[4]*v*t**params[5]+params[6]*x