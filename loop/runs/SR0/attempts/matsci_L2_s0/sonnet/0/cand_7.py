def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*eps+params[3]*(eps*T)**params[4]+params[5]*T**2