def equation(t,x,v,params):
    return params[0]*np.sin(params[1]*t+params[2])+params[3]+params[4]*x+params[5]*v+params[6]*np.sin(2*params[1]*t+params[7])+params[8]*x**2+params[9]*v**2