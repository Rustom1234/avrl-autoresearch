def equation(t,P,params):
    return params[0]*P**(-params[1]) + params[2]*P + params[3]*t + params[4]*t*P**(-params[1])+params[5]