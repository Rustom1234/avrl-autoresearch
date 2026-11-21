def equation(t, A, params):
    return -(params[0]*A*t**params[1] + params[2]*A**2*t**params[3])