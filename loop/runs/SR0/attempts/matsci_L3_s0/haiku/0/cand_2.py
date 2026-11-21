def equation(eps, T, params):
    s0 = params[0] + params[1]*T
    k = params[2] + params[3]*T
    return s0 + k*(1-np.exp(-params[4]*eps)) + params[5]*eps
