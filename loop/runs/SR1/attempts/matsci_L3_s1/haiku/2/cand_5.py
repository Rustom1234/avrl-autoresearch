def equation(eps, T, params):
    return params[0] + params[1]*eps**params[2]*np.exp(-params[3]*eps) + (params[4]*T**2 + params[5]*T + params[6]) * (1 + params[7]*eps**params[8]*np.exp(-params[9]*eps))
