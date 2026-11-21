def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*x**3+params[3]*v+params[4]*np.cos(params[5]*t+params[6])+params[7]*v**2