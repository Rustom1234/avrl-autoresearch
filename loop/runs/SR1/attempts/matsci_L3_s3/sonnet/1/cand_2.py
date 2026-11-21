def equation(eps,T,params):
    return (params[0]+params[1]*eps**params[2])*(T+params[3])**params[4]+params[5]