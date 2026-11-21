def equation(t, A, params):
    return params[0]*A**params[1]*np.log(1+params[2]*t)