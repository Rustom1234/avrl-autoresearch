def equation(t, A, params):
    return params[0] + params[1]*A**params[2] + params[3]*np.tanh(params[4]*t)/A + params[5]*np.log(A)+params[6]*A*t