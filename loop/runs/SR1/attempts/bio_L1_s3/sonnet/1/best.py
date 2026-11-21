def equation(t,P,params):
    return params[0]*P**params[1]*(1-(P/params[2])**params[3])