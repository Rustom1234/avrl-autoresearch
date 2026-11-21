def equation(eps, T, params):
    return params[0] + params[1]*T**2 + params[2]/T + (params[3] + params[4]*T**2)*eps**params[5] + params[6]*eps/T + params[7]*eps