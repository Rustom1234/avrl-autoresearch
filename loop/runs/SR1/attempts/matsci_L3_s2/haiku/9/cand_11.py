def equation(eps, T, params):
    return params[0] + params[1]*np.exp(-params[2]/T) + params[3]*eps**params[4] + params[5]/eps + params[6]*eps**2 + params[7]*eps*np.exp(-params[2]/T) + params[8]*T
