def equation(eps, T, params):
    return params[0] + params[1]*eps**params[2] + params[3]*T + params[4]*T**2 + params[5]*eps*T