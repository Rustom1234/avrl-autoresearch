def equation(t, A, params):
    k = params[0] + params[1]*t + params[2]*t**2
    return k*A/(1 + params[3]*A) + params[4]*A*t