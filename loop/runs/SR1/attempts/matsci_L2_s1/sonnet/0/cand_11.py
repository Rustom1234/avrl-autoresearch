def equation(eps,T,params):
    t=T-4
    return params[0]+params[1]*t+params[2]*t**2+params[3]*t**3+eps*(params[4]+params[5]*t+params[6]*t**2+params[7]*t**3)