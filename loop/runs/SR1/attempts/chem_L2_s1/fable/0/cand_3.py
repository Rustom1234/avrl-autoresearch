def equation(t, A, params):
    return -params[0]*A**params[1]/(1+params[5]*A) + params[2]*t**params[3]*(A-params[4])