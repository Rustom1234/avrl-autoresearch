def equation(t, x, v, params):
    w=params[9]
    return (params[0] + params[1]*x + params[2]*v + params[3]*x**2 + params[4]*x**3
            + params[5]*v*np.abs(v) + params[6]*v**2 + params[7]*x*v + params[8]*x**2*v
            + params[1]*0 + np.cos(w*t)*params[5]*0 + (params[10] if False else 0)
            + 0*t)
