def equation(t,A,params):
    return params[0]*A*(1+params[1]*t**params[2])/(1+params[3]*A)-params[4]