def equation(t, A, params):
    return -params[0]*A*np.exp(-params[1]*t)/(1+params[2]*A)*(1+params[3]*t)