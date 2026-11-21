def equation(t, A, params):
    return params[0]*np.log(A+params[1]) + (params[2] + params[3]*t**params[4])/A**params[5] + params[6]