def equation(t,P,params):
    return params[0]+params[1]*P+params[2]/(P+params[3])+params[4]*t**params[5]*P**params[6]