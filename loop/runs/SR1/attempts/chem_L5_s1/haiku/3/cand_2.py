def equation(t, A, params):
    return -params[0]*np.exp(params[2]*t)*A**params[1]