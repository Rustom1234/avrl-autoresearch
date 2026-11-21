def equation(t, x, v, params):
    return (params[0] + params[1]*t + params[2]*np.sqrt(v) + params[3]*x + params[4]*x**2 + params[5]*x**3
            + params[6]*t*x**2 + params[7]*np.cos(v) + params[8]*t*np.exp(-x) + params[9]*v)