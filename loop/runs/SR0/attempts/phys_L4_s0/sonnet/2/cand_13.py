def equation(t,x,v,params):
    return params[0]+params[1]*v+params[2]*np.sin(params[3]*x)+params[4]*t**params[5]*x**params[6]*v**params[7]+params[8]*np.cos(params[9]*t)