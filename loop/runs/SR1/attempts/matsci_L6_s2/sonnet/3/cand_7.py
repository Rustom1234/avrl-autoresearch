def equation(eps,T,params):
    return (params[0]*T**params[1]+params[2]*eps**params[3])*(1+params[4]*eps)