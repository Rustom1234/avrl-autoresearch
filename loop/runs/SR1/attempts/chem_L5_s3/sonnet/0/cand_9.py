def equation(t, A, params):
    return params[0]+params[1]*np.log(A+params[2])+params[3]*t**params[6]*np.exp(-params[4]*t)