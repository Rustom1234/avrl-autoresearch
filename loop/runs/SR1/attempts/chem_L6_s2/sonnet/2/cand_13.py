def equation(t, A, params):
    return -params[0]/A - params[1]*(1-np.exp(-params[2]*A)) + params[3]*t**2/A