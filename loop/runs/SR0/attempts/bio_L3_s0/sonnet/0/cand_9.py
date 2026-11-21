def equation(t,P,params):
    return params[0]*P**(-0.5)*(1+params[1]*t)+params[2]*P+params[3]+params[4]*t+params[5]*P*t