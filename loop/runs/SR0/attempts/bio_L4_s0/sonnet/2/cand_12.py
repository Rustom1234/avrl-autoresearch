def equation(t,P,params):
    return params[0]*P**params[1]*(1+params[2]*t)+params[3]*P*t+params[4]*t**2+params[5]*t+params[6]*P+params[7]