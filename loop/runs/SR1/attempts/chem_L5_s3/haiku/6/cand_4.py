def equation(t, A, params):
    return params[0]*A/(1+params[1]*A)*t**params[2]*np.exp(-params[3]*t)
