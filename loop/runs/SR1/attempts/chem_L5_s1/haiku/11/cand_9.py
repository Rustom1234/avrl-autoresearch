def equation(t, A, params):
    return -params[0]*t**params[1]*A*np.exp(-params[2]*A)
