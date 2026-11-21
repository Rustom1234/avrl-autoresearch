def equation(eps, T, params):
    n=params[9]
    return params[0] + params[1]*T + params[2]*T**2 + params[3]*eps**n + params[4]*eps*T + params[5]*eps**n*T + params[6]*eps**2 + params[7]*eps**n*T**2 + params[8]*eps*T**2