def equation(t, A, params):
    return params[0]+params[1]*A**params[2]+params[3]*np.sin(params[4]*t+params[5])+params[6]*t