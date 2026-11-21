def equation(eps,T,params):
    return params[0]+params[1]/T+params[3]*eps**3/T+params[6]*T**params[4]+params[5]*eps**3*T**params[7]