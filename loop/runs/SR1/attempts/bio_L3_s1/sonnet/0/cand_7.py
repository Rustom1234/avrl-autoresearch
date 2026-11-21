def equation(t,P,params):
    return params[0]*P+params[1]/P+params[2]*t*P**params[3]+params[4]*t+params[5]