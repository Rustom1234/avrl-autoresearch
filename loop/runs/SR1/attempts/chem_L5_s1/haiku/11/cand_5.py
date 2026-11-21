def equation(t, A, params):
    return -(params[0] + params[1]*t)*A/(1+params[2]*A)
