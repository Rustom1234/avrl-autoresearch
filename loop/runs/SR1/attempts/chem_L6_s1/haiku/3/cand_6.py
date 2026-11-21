def equation(t, A, params):
    k = params[0] + params[1]*np.cos(params[2]*t + params[3])
    return k*A**params[4] + params[5]*A**params[6]