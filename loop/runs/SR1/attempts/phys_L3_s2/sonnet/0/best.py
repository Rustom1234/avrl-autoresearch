def equation(t,x,v,params):
    return params[0]+params[1]*v*x**params[2]+params[3]*x+params[4]*np.log(x)+params[5]*np.cos(params[6]*t+params[7])+params[8]*v**params[9]