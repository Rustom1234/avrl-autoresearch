def equation(t,x,v,params):
    return params[0]+params[1]*t+params[2]*np.log(x)+params[3]*v**params[4]+params[5]*x+params[6]*np.sin(params[7]*t)