def equation(t, A, params):
    return params[0]*A**2*np.exp(-params[1]*t*A**params[2]) + params[3]*t*A**params[4] + params[5]*A**params[6]