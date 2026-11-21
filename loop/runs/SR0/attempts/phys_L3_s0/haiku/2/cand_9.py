def equation(t, x, v, params):
    w=params[0]
    return (params[1] + params[2]*x + params[3]*x**2 + params[4]*x**3 + params[5]*v + params[6]*v*np.abs(v)
            + params[7]*v**2 + params[8]*x*v + params[9]*np.cos(w*t) + 0*t) + params[9]*0 + 0*x + (0)*v
