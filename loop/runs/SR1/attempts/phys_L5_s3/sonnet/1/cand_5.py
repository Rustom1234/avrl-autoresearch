def equation(t,x,v,params):
    return params[0]*np.sin(params[1]*x)+params[2]*np.cos(params[3]*t)+params[4]*v+params[5]