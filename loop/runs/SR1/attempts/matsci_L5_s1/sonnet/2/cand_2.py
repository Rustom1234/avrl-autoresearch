def equation(eps,T,params):
    return params[0]*(1-T/params[1])*eps**params[2]+params[3]*T+params[4]