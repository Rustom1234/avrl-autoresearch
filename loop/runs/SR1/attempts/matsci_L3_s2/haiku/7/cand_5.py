def equation(eps, T, params):
    return params[0] + params[1]*np.exp(-params[2]*eps) + params[3]*eps**params[4] + params[5]*np.exp(-params[6]*T) + params[7]*T + params[8]*eps*T
