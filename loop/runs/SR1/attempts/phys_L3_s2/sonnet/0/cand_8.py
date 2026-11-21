def equation(t,x,v,params):
    return params[0]+params[1]*v*(1+params[2]*x**2)+params[3]*x**params[4]+params[5]*np.sin(params[6]*t+params[7])+params[8]*x+params[9]*v**2