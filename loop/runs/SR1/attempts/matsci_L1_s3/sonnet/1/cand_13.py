def equation(eps,T,params):
    return -(params[0]*eps**1.5+params[1]*T**params[2]+params[3]+params[4]*eps*T)