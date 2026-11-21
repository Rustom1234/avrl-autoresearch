def equation(eps,T,params):
    return params[0]*eps**(params[1]-params[2]*T**params[6])+params[3]+params[4]*T**params[5]