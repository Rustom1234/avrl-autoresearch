def equation(t, A, params):
    return -(params[0]+params[1]*t)*(params[2]+A)/(1+params[3]*A**2)