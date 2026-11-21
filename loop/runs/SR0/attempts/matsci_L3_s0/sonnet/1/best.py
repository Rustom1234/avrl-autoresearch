def equation(eps,T,params):
    return params[0]*eps**params[1]*T**params[2]+params[3]*(1-(T/params[4])**params[5])*eps**params[6]