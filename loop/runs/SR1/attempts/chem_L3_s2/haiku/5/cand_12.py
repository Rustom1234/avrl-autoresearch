def equation(t, A, params):
    g = 1 - np.exp(-params[4]*t)
    return params[0] + params[1]*A + params[2]*g + params[3]*A**2*g
