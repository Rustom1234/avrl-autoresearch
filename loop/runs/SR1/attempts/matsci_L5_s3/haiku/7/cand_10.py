def equation(eps, T, params):
    return params[0] + params[1]*T**params[2] + (params[3] + params[4]*T)*eps**params[5] + params[6]*eps**2 + params[7]*T**2