def equation(eps, T, params):
    return -(params[0]*T + params[1]*T**2 + params[2]*T**3) + params[3]*eps**params[4]