def equation(t, A, params):
    g = 1 - np.exp(-params[4]*t)
    return params[0]*A + params[1]*A**2 + params[2]*g + params[3]*A**2*g
