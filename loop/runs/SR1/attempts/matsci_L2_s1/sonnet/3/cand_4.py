def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T**2+eps**params[3]*(params[4]+params[5]*T+params[6]*T**2)