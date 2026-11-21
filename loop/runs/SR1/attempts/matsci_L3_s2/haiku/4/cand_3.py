def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]*(1-np.exp(-params[3]*eps))