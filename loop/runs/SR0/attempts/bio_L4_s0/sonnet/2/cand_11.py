def equation(t,P,params):
    return params[0]*P**params[1]+params[2]*P*t+params[3]*t**2+params[4]*t+params[5]*P+params[6]