def equation(eps, T, params):
    A = params[1] + params[2]*T + params[3]*T**2
    return params[0] + A*eps**params[4]*np.exp(-params[5]*eps)
