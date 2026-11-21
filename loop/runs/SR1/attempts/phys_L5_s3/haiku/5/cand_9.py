def equation(t, x, v, params):
    return (params[0] + params[1]*t)*x**params[2]*np.exp(-params[3]*x) + params[4] + params[5]*v