def equation(eps,T,params):
    return params[0]*eps+params[1]*eps**(params[2]*(T-params[3]))+params[4]*(T-params[3])**2+params[5]*eps*(T-params[3])