def equation(eps, T, params):
    return params[0] + params[1]*T**params[2] + (params[3] + params[4]*T**params[5])*eps**2 + params[6]*eps**4 + params[7]*T*eps**4
