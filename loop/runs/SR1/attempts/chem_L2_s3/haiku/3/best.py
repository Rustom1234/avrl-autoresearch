def equation(t, A, params):
    return -(params[0] + params[1]*A**params[2])*(1+params[3]*t+params[4]*t**2)*(1+params[5]*A)
