def equation(t, A, params):
    return params[0]+params[1]*A**params[2]+params[3]*t*np.exp(-params[4]*t)