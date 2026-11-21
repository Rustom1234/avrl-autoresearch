def equation(t,P,params):
    return params[0]+params[1]*P+params[2]*P**2+params[3]*t/(t+params[4])+params[5]*P*t/(t+params[4])+params[6]*t