def equation(t,P,params):
    return params[0]*P*(1-P/params[1])-params[2]*P/(P**2+params[3])+params[4]*P**params[5]