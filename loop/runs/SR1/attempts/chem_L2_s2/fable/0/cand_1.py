def equation(t, A, params):
    return params[0] + params[1]*A**params[2] + params[3]*t**params[4]*np.exp(-params[5]*A) + params[6]*A/(params[7]+A)