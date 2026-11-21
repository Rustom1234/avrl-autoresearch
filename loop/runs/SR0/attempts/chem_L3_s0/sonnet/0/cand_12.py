def equation(t, A, params):
    return -params[0]*t**params[5]+params[1]*A**params[2]/(params[3]+A**params[2])*t**params[4]