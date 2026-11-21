def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) + params[2] + params[3]*A + params[4]*A**2 + params[5]*A**3 + params[6]*t*A
