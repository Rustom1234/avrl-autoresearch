def equation(eps,T,params):
    return params[0]*T+params[1]*eps**params[2]*T**params[3]-params[4]*eps**params[5]*T**params[6]