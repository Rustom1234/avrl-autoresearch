def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*x**3+params[3]*np.log(x)+t*(params[4]+params[5]/x)+params[6]*v+params[7]*x**2