def equation(t, A, params):
    return -params[0]*A**params[1]/(params[2]+A**params[1]) + (params[3]+params[4]*t**2)/A + params[5]