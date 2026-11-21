def equation(t,P,params):
    return params[0]*t**params[1]+P**params[2]*(params[3]+params[4]*t**params[5])