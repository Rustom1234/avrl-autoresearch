def equation(eps,T,params):
    return params[0]*eps**params[1]*(1-T/params[2]) + params[3] + params[4]*T**params[5]