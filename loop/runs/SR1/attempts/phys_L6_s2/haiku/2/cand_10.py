def equation(t, x, v, params):
    return (params[0] + params[1]*t + params[2]*v + params[3]*x + params[4]*x**2 + params[5]*x**3
            + params[6]*t*np.exp(-x) + params[7]*np.sqrt(v) + params[8]*np.sin(x) + params[9]*np.cos(v))