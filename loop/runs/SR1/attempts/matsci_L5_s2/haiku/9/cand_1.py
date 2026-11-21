def equation(eps, T, params):
    return params[0]*eps**params[1]*np.exp(-params[2]*eps/T)+params[3]*eps