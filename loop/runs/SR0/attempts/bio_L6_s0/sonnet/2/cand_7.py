def equation(t,P,params):
    return params[0]*P+params[1]*t**params[2]*(1+params[3]*P**params[4])+params[5]*P**2