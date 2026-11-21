def equation(t, A, params):
    return -params[0]*(1+params[1]*t**params[2])*A/(params[3]+A+params[4]*A**2)