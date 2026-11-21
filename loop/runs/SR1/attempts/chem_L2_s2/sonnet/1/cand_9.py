def equation(t, A, params):
    return params[0] + params[1]*A**params[2] + params[3]*np.sqrt(t)*A**(-params[4]) + params[5]*np.exp(-params[6]*A)