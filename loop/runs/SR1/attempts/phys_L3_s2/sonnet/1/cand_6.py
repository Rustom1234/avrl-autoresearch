def equation(t,x,v,params):
    return params[0]+params[1]*np.log(x)+params[2]*v+params[3]*np.cos(params[4]*t+params[5])+params[6]*x