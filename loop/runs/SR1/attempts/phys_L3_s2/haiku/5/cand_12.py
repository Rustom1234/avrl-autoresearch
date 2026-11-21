def equation(t, x, v, params):
    w=params[0]
    return (params[1]*np.cos(w*t)+params[2]*np.sin(w*t)+params[3]
            + params[4]*x + params[5]*np.abs(x)**params[6]
            + params[7]*v + params[8]*np.abs(v)**params[9]
            + 0.0*x)
