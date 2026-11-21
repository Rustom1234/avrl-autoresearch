def equation(t, A, params):
    return -(params[0]*A**params[1] + params[2]*t**3*(params[3]-A))
