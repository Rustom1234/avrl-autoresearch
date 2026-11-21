def equation(eps, T, params):
    return params[0]*eps**2/(1+params[1]*eps**2) + params[2]*T**3 + params[3]*eps**3*T**2