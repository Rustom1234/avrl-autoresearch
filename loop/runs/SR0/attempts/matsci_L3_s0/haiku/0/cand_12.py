def equation(eps, T, params):
    return params[0]*eps**params[1]*T**params[2] + params[3]*eps + params[4]*eps**2 + params[5]*T**3 + params[6]*T + params[7] + params[8]*eps*T + params[9]*eps**3
