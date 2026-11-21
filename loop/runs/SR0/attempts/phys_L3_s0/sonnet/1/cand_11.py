def equation(t,x,v,params):
    return params[0]*x+params[1]*x**3+params[2]*v**params[3]+params[4]*x*t**params[5]+params[6]*np.cos(params[7]*t)+params[8]