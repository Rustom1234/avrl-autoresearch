def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*eps**1.5*(1+params[3]*T)+params[4]*T**2+params[5]*eps