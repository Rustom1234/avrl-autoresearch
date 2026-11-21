def equation(eps, T, params):
    return params[0]+params[1]*np.sqrt(eps)+params[2]*np.exp(-T/params[3])