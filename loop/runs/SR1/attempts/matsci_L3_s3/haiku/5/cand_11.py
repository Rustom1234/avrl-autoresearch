def equation(eps, T, params):
    return params[0] + params[1]*T**2 + (params[2] + params[3]*T**2)*eps**params[4] + params[5]*eps*T + params[6]*eps*T**2