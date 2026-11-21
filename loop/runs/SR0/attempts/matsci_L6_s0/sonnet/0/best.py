def equation(eps,T,params):
    return params[0]*eps**params[1]+params[2]*T**3+params[3]*eps*T+params[4]*eps+params[5]*T+params[6]