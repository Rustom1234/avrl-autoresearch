def equation(eps, T, params):
    return params[0]*np.abs(params[1]-eps)**params[2] + params[3] + params[4]*T
