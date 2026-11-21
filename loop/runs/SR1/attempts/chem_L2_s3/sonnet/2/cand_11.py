def equation(t, A, params):
    return params[0] + params[1]*np.log(t) - params[2]*A**params[3]*t**params[4] - params[5]*A**params[6]/t**params[7]