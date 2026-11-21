def equation(eps, T, params):
    return (params[0] + params[1]*T**params[2]) + (params[3] + params[4]*T + params[5]*T**2)*eps**2 + params[6]*eps**3 + params[7]*eps