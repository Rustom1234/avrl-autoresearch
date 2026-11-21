def equation(t, A, params):
    return params[0]*t*np.exp(-params[1]*t) + params[2]*np.exp(-params[3]*t)*A + params[4]*A*t**2 + params[5]*A**2*t**params[6]+ params[7]*A**2