def equation(t,x,v,params):
    return params[0]*x**params[1]*np.exp(-params[2]*x)+params[3]*np.cos(params[4]*t)+params[5]*v+params[6]