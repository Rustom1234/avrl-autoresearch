def equation(t, x, v, params):
    w = params[8]
    return (params[0] + params[1]*t + params[2]*x + params[3]*v + params[4]*x**3
            + params[5]*v*np.abs(v) + params[6]*t*x + params[7]*np.cos(w*t) + params[9]*np.sin(w*t))
