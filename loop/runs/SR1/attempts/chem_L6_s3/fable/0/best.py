def equation(t, A, params):
    return params[0]*np.exp(-params[1]*np.sqrt(t)) - params[2]/t - params[3]*A**1.5*t**2