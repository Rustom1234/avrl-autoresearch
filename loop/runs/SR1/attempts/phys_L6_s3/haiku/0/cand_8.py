def equation(t, x, v, params):
    w=params[9]
    return (params[0] + params[1]*x + params[2]*x**3 + params[3]*v + params[4]*v*np.abs(v)
            + np.cos(w*t)*(params[5]+params[6]*v+params[7]*x) + np.sin(w*t)*(params[8]*v))