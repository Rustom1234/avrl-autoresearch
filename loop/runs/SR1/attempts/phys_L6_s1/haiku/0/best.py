def equation(t, x, v, params):
    w=params[9]
    return (params[0] + params[1]*v + params[2]*x + params[3]*x**3 + params[4]*v*np.abs(v)
            + (params[5] + params[6]*v)*np.cos(w*t) + (params[7] + params[8]*v)*np.sin(w*t))