def equation(t, A, params):
    e = np.exp(params[2]*t)
    return (params[0]+params[1]*e)*A + params[3]*A**2 + (params[4]+params[5]*e)/A + params[6]+params[7]*e