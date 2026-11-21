def equation(t,P,params):
    return params[0] + params[1]*P + params[2]*P**2 + params[3]/P**params[4] + params[5]*t*P + params[6]*t*P**2 + params[7]*t + params[8]*t**2 + params[9]*t*t*P
