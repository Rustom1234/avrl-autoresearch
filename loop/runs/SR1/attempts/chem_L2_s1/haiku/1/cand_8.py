def equation(t, A, params):
    return -params[0]*A/(1+params[1]*A*np.exp(params[2]*t))*np.exp(params[3]*t)