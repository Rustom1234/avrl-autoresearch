def equation(t,P,params):
    return params[0]*P**params[1]+params[4]+t*(params[2]+params[3]*P**params[5])