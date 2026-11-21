def equation(t,x,v,params):
    return params[0]*np.sin(params[1]*t**params[2]+params[3])+params[4]+params[5]*t+params[6]*x+params[7]*v