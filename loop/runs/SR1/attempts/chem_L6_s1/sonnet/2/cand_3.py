def equation(t, A, params):
    return params[0]*A/(1+params[1]*A)*np.cos(params[2]*t)+params[3]*t