def equation(t, x, v, params):
    w=0.5
    return (params[0] + params[1]*np.exp(-params[2]*x) + params[3]*x + params[4]*np.tanh(params[5]*v)
            + params[6]*np.cos(w*t) + params[7]*np.sin(w*t) + params[8]*v*np.cos(w*t) + params[9]*v*np.sin(w*t))