def equation(t, P, params):
    return params[0]*P/(1+params[1]*P)*(1-P/params[2]) + params[3]*t*P
