def equation(t,P,params):
    return params[0]*t**params[1]+P*(params[2]+params[3]*t**params[4])-params[5]*P**2