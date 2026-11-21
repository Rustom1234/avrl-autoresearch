def equation(t,P,params):
    return params[0] + params[1]*P + params[2]*P**(-params[5]) + params[3]*t*P*(1-P/params[4])