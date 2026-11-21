def equation(t,x,v,params):
    return params[0]-params[1]*x-params[2]*x**3-params[3]*v**params[4]+params[5]*np.sin(params[6]*t+params[7])+params[8]*t