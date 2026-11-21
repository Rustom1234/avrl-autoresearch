def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*x**2+params[3]/x+params[4]*x/v+params[5]*np.cos(x)+params[6]*np.sin(params[7]*t)+params[8]*np.cos(params[7]*t)+params[9]*x**3