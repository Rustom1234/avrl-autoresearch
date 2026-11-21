def equation(t, P, params):
    return params[0]*P+params[1]*P**2 + params[2]/(1+params[3]*P**params[5])+params[4]