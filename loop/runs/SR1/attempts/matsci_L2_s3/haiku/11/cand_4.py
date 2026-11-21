def equation(eps, T, params):
    return (params[0] + params[1]*eps**params[2]) * (1 + params[3]*T**params[4]) * (1 + params[5]*eps**params[6]*T**params[7])