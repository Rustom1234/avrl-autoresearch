def equation(t,P,params):
    return params[0]*P**(-params[1])+params[2]*t+params[3]*t*P**params[4]+params[5]