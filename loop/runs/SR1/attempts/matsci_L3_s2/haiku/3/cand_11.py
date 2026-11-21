def equation(eps, T, params):
    g = params[0] - params[1]*np.exp(-params[2]*T)
    k = params[3] + params[4]*np.exp(-params[5]*T)
    return g - k*np.log(1 + params[6]*eps) + params[7]*eps*np.exp(-params[8]*T) + params[9]*eps
