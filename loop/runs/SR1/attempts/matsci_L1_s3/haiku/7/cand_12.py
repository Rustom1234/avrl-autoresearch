def equation(eps, T, params):
    return -(params[0]*eps**params[1]*(1+params[2]*T**2) + params[3]*(T-params[4])**2 + params[5]*eps*T + params[6]*eps**2)