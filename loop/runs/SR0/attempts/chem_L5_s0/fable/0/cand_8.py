def equation(t, A, params):
    return params[0]+params[1]*A+params[2]*A**params[3]*np.exp(-params[4]/A**params[5])