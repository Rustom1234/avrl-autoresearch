def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)/A**2 + params[2]*t**2/A + params[3]*A**2*np.exp(-params[4]*t) + params[5]*A + params[6] + params[7]*np.sqrt(A)