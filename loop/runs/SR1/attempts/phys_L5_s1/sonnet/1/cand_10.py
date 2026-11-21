def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*v**params[3]+params[4]*np.cos(params[5]*t+params[6])+params[7]*x**3+params[8]*v