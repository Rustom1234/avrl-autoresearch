def equation(t,P,params):
    return params[0]*P+params[1]*P**2+params[2]*t*P/(params[3]+P)+params[4]*t*P