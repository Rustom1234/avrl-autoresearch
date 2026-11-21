def equation(t, A, params):
    return params[0]*np.sqrt(A)*t**params[1]*np.exp(-params[2]*t) + params[3]*t**params[4]*np.exp(-params[5]*t)