def equation(t,x,v,params):
    return params[0]*x+params[1]/v+params[2]*x**params[3]*v**params[4]+params[5]+params[6]*np.sin(params[7]*t)+params[8]*np.cos(params[7]*t)