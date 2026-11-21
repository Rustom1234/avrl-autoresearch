def equation(eps, T, params):
    return np.exp(params[0] + params[1]*T + params[2]*eps + params[3]*T*eps) * eps**(params[4] + params[5]*T)