def equation(t,P,params):
    return P**params[0]*(params[1]+params[2]*t+params[3]*P+params[4]*t*P+params[5]*t**2)