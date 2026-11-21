def equation(t,P,params):
    return params[0]*P+params[1]*P**params[2]*(1+params[3]*t)+params[4]*P*t