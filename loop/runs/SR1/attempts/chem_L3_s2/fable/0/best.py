def equation(t, A, params):
    return params[0]*np.sqrt(t) + params[1]*t*np.log(t) + params[2]*A**3*np.exp(-params[3]*t)