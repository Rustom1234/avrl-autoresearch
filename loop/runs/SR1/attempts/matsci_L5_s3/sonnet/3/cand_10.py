def equation(eps,T,params):
    return params[0]+params[1]*T/(1+params[2]*T)+params[4]*eps**params[3]