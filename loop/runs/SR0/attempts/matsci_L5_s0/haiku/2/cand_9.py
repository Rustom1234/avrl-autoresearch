def equation(eps, T, params):
    return params[0]*np.exp(-params[1]*eps) + params[2] + params[3]*eps + params[4]*T + params[5]*T*eps
