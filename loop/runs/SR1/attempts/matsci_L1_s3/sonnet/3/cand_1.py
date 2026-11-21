def equation(eps,T,params):
    return -(params[0]*T**params[1] + params[2]*eps**params[3]*T**params[4]*0+params[2]*eps**params[3]*(1+params[5]*T))