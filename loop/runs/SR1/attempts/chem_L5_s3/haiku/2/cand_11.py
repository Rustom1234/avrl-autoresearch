def equation(t, A, params):
    return (params[0]*(1-np.exp(-params[1]*A))+params[4])*t**params[2]*np.exp(-params[3]*t)