def equation(t,x,v,params):
    return params[0]*x+params[1]*x*t+params[2]*v+params[3]/x+params[4]+params[5]*np.sin(params[6]*t+params[7])+params[8]*x**3