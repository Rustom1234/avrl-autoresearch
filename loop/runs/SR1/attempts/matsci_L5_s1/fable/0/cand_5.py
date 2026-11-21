def equation(eps, T, params):
    return params[0]+params[1]*np.sqrt(eps)+params[2]*np.abs(T+params[4])**params[3]