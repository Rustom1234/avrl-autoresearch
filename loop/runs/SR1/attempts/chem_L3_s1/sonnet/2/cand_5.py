def equation(t, A, params):
    return params[0]*np.cos(params[1]*t+params[5]) + params[2]*A + params[3]*A**2 + params[4]*A**3