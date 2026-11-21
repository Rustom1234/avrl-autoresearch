def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*x**2+params[3]/x+params[4]*x**params[6]*v**(-params[5])+params[7]*np.cos(x)