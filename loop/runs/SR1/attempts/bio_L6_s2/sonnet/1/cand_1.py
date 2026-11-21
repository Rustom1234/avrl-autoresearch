def equation(t,P,params):
    return -P*(params[0]+params[1]*P)*(1+params[2]*t)+params[3]*P**2