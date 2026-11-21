def equation(t,x,v,params):
    return params[0]+params[1]*v*x**params[2]+params[3]*x+params[4]*x**params[5]+params[6]*np.cos(params[7]*t+params[8])