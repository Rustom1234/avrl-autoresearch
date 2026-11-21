def equation(t, x, v, params):
    w = params[1]
    return (params[0]*np.sign(v) + params[2]*np.cos(w*t) + params[3]*np.sin(w*t)
            + params[4]*v + params[5]*x + params[6]*x**3 + params[7]*np.abs(v)*v
            + params[8]*np.cos(2*w*t) + params[9]*np.sin(2*w*t))