def equation(t, x, v, params):
    return params[0] + params[1]*np.sin(params[2]*x) + params[3]*x*np.exp(-params[4]*x**2) + params[5]*t + params[6]*t**2 + params[7]*x**2 + params[8]*x**3