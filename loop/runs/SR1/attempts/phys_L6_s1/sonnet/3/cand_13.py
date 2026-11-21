def equation(t,x,v,params):
    return params[0]+params[1]*t+params[2]*np.sin(params[3]*t+params[4])+v*(params[5]+params[6]*np.sin(params[7]*t+params[8]))-params[9]*x