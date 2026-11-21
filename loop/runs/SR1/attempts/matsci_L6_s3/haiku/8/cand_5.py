def equation(eps, T, params):
    return params[0]*eps**(params[1]+params[2]/T) + params[3]*T + params[4]*eps
