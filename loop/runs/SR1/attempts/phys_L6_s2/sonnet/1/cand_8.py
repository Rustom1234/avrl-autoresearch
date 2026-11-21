def equation(t,x,v,params):
    return params[0]+params[1]*np.sqrt(t)*np.sin(params[2]*t+params[3])+params[4]*x+params[5]*x**3+params[6]*v