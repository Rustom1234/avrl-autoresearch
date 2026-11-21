def equation(t, A, params):
    k = params[0]+params[1]*t
    return -k*A**params[2]/(1+params[3]*A)**params[4]
