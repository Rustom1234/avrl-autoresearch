def equation(t,x,v,params):
    return params[0]+params[1]*v+params[2]*x+params[3]*np.sin(params[4]*t+params[5])+params[6]*x**params[7]