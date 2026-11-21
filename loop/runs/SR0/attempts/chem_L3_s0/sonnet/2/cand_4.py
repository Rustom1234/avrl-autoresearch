def equation(t, A, params):
    return params[0] + params[1]*A + params[2]*np.sqrt(t)*A + params[4]*A**2+params[3]*np.log(t)