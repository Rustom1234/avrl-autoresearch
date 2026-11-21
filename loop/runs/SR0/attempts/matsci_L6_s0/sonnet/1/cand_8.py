def equation(eps,T,params):
    return params[0]+params[1]*T+params[5]*T**2+params[6]*T**3+(params[2]+params[3]*T+params[7]*T**2)*eps**params[4]