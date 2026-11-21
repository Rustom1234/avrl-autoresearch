def equation(t, A, params):
    return params[0] + params[1]*A**params[2] + params[3]*t/(1+params[4]*t)*A**params[5]