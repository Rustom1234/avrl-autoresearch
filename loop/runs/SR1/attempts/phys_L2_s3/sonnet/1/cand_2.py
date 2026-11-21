def equation(t,x,v,params):
    return params[0]*x+params[1]*x**3+params[2]*v+params[3]*np.cos(params[4]*t+params[5])+params[6]+params[7]*v**2