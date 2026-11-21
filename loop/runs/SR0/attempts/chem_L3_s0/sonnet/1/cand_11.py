def equation(t,A,params):
    return params[0]*A*(1+params[1]*t**params[2])+params[3]*A**2-params[4]