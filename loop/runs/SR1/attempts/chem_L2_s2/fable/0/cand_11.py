def equation(t, A, params):
    return params[0]*np.sqrt(t)/A + params[1]*A**2 - params[2]/(params[3]+A)