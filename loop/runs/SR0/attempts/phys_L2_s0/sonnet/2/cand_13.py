def equation(t,x,v,params):
    return params[0]+params[1]*v+params[2]*x*v+params[3]/t+params[4]*v**params[5]+params[6]/x+params[7]*np.cos(params[8]*t)