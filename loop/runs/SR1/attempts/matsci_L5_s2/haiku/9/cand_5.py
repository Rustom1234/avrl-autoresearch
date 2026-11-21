def equation(eps, T, params):
    return params[0]*eps**params[1]/(1+params[2]*eps**params[3])*(1+params[4]*T)