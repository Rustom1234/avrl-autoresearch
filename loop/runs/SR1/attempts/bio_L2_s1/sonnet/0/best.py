def equation(t,P,params):
    return params[0]+params[1]*P**params[2]+params[3]*P+t*(params[4]+params[5]*P)+params[6]*t*P**params[2]