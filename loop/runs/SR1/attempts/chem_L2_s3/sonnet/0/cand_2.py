def equation(t, A, params):
    return -(params[0]/t + params[1] + params[2]*A**params[3]*(1+params[4]*t))