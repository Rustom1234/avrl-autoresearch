def equation(eps, T, params):
    return params[0] + params[1]*T**2 + params[2]/T + (params[3] + params[4]*T**2 + params[5]/T)*eps**params[6] + params[7]*eps/T