def equation(eps,T,params):
    return params[0]*T**3+params[1]*eps**(params[2]+params[3]*T)