def equation(t, A, params):
    return params[0]+params[1]*A+params[2]*np.log(A)+params[3]*np.sqrt(t)/A**params[4]