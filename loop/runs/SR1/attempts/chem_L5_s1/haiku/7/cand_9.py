def equation(t, A, params):
    return -params[0]*np.exp(params[1]*t)*A**params[2]/(1+params[3]*A)**params[4]
