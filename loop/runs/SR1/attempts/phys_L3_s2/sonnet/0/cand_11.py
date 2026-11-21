def equation(t,x,v,params):
    return params[0]+params[1]*v+params[2]*x+params[3]*np.exp(-params[4]*x)+params[5]*np.cos(params[6]*t+params[7])+params[8]*v*x**2+params[9]*v**2