def equation(t, x, v, params):
    w=params[4]
    return params[0] + params[1]*x + params[2]*v + params[3]*np.cos(w*t) + params[5]*np.sin(w*t) + params[6]*x**2 + params[7]*x**3 + params[8]*v**2
