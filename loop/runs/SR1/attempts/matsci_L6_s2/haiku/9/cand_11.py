def equation(eps, T, params):
    return params[0]*np.exp(params[3]*T)*eps**(params[1]+params[2]*T)