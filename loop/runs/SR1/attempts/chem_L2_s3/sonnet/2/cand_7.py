def equation(t, A, params):
    return -params[0]/t**params[1] - params[2]*A**params[3]*t**params[4] - params[5]*A**params[6] - params[7]*A**params[8]/t**params[9]