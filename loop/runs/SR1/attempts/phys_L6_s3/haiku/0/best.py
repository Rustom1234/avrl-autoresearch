def equation(t, x, v, params):
    w=params[6]
    return (params[0] + params[1]*np.exp(-params[2]*x) + params[3]*np.tanh(params[4]*v)*(1+params[5]*np.cos(w*t))
            + params[7]*x + params[8]*np.cos(w*t) + params[9]*np.sin(w*t))