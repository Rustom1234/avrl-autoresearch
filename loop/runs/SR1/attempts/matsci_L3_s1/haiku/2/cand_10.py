def equation(eps, T, params):
    A = params[0] + params[1]*T + params[2]*T**2
    return A*eps*np.exp(-params[3]*eps) + params[4] + params[5]*T + params[6]*T**2
