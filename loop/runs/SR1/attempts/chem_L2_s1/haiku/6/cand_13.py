def equation(t, A, params):
    return -(params[0]*A/(1+params[1]*A) + params[2]*t**params[3]*(params[4]-A))
