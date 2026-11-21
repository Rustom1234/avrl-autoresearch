def equation(t, A, params):
    return -params[0]*np.exp(params[1]*t)*A/(1+params[2]*np.exp(params[3]*t)*A)
