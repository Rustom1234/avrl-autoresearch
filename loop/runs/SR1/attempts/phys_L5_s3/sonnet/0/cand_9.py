def equation(t,x,v,params):
    return params[0]+params[1]*np.sin(params[2]*x+params[3])+params[4]*x**-0.5+params[5]*np.cos(params[6]*t+params[7])+params[8]*v**params[9]