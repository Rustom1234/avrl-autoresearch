def equation(eps, T, params):
    return params[0] + params[1]*eps**params[9] + eps**2*(params[2]*np.abs(T+params[3])**params[4] + params[8]) + params[5]*np.tanh(params[6]*T+params[7])