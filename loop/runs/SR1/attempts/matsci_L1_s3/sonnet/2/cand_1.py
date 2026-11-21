def equation(eps,T,params):
    return -(params[0]*(T-2)**params[1] + params[2]*eps**params[3]*T**params[4])