def equation(t, x, v, params):
    w = params[1]
    return (params[0]*np.sign(v) + params[2]*np.cos(w*t) + params[3]*np.sin(w*t)
            + params[4]*np.cos(2*w*t) + params[5]*np.sin(2*w*t) + params[6]*v + params[7]*x)