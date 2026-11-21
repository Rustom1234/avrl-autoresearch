def equation(t, A, params):
    return -params[0] - params[1]*A**2*np.sqrt(t) - params[2]/t - params[3]*A*np.exp(-params[4]*t)