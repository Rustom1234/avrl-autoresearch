def equation(t, A, params):
    sat = (1 - params[1]*np.exp(-params[2]*t))
    return (params[0] + params[3]*A**params[4])*sat + params[5]*A**params[6]*np.exp(-params[7]*t)