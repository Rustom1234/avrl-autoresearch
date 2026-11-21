def equation(t,P,params):
    return params[0]/P + params[1]*P*(1-P/params[2]) + params[3] + (params[4]*t+params[5])*P/(1+params[6]*P)