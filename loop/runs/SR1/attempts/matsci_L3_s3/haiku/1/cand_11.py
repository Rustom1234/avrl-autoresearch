def equation(eps, T, params):
    return np.exp(params[0] + params[1]*T + params[2]*T**2) * eps**(params[3] + params[4]*T + params[5]*T**2)