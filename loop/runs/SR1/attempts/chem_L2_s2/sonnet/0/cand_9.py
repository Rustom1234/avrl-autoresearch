def equation(t, A, params):
    return params[0] + params[1]*A + params[2]*np.sqrt(t)*np.exp(-params[3]*A) + params[4]*np.sin(params[5]*A)