def equation(t, A, params):
    return -params[0]*np.abs(A)**params[1]*np.exp(params[2]*t)