def equation(eps, T, params):
    return params[0] + params[1]*np.log(T) + params[2]*np.abs(eps)**3/T + params[3]*np.abs(eps)**params[4]/T
