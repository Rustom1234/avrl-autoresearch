def equation(t, x, v, params):
    w=0.5
    return (params[0] - params[1]*x - params[2]*np.abs(x)**params[3]*np.sign(x)
            - params[4]*np.sign(v)*np.abs(v)**params[5]
            + params[6]*np.cos(w*t) + params[7]*np.sin(w*t))