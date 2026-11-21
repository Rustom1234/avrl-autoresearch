def equation(t, A, params):
    return params[0]*np.log(A/params[1])*(1-np.exp(-params[2]*t))+params[3]*A*t*np.exp(-params[4]*t)