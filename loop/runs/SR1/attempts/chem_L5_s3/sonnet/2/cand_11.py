def equation(t, A, params):
    return params[0]*np.sqrt(A)+params[1]*np.sin(params[2]*t+params[3])+params[4]+params[5]*t