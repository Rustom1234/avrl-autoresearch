def equation(t,P,params):
    return params[0]/P + params[1]*P + params[2]*t*P/(1+params[3]*P**2) + params[4]