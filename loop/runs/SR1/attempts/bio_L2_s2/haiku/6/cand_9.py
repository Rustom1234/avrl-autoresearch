def equation(t,P,params):
    r=params[0]+params[1]*t
    return r*P*(1-(P/params[2])**params[3]) + params[4] + params[5]*t