def equation(eps, T, params):
    return params[0]*np.exp(params[1]*T) + params[2]*np.exp(params[4]*T)*eps**(params[3]+params[5]*T)