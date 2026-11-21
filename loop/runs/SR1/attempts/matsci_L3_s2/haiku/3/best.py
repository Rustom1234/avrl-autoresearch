def equation(eps, T, params):
    g = params[0] - params[1]*np.exp(-params[2]*T)
    a = params[3] + params[4]*np.exp(-params[5]*T)
    return g + a*np.exp(-params[6]*eps) + params[7]*eps + params[8]*eps**2 + params[9]*T*eps
