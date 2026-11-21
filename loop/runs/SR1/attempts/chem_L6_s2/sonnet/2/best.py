def equation(t, A, params):
    return (params[0] + params[1]*np.sin(params[2]*A+params[3]) + params[4]*t**2)/A