def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*t**2 + (params[3] + params[4]*t + params[5]*t**2)*A**params[6]
