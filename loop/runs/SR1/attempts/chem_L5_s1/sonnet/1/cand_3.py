def equation(t, A, params):
    return -params[0]*t**params[2]*A**params[5]/(params[3]+A**params[6])**params[4]