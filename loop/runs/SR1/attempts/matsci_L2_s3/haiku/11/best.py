def equation(eps, T, params):
    return params[0] + params[1]*T**params[2] + (params[3] + params[4]*T**params[5])*eps**(params[6] + params[7]*T) + params[8]*eps**2*T**params[9]