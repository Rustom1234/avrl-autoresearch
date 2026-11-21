def equation(eps, T, params):
    return -(params[0] + params[1]*T)*eps**(params[2]) * np.exp(params[3]*T)
