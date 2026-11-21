def equation(t, x, v, params):
    w = params[0]
    return (params[1] + params[2]*np.cos(w*t) + params[3]*np.sin(w*t)
            + params[4]*np.cos(2*w*t) + params[5]*np.sin(2*w*t)
            + params[6]*x + params[7]*v + params[8]*x**3 + params[9]*v*np.abs(v))