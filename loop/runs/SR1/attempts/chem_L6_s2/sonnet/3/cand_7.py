def equation(t, A, params):
    return params[0]+params[1]/A+params[2]*A**params[3]+params[4]*t**params[5]/A**params[6]