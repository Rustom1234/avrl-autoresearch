def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T**2+params[3]*eps*T**params[4]+params[5]*eps**2*T**params[6]+params[7]*eps