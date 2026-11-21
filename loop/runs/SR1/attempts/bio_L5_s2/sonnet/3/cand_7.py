def equation(t,P,params):
    return params[0]*(1-(P/params[1])**params[2])+params[3]*P