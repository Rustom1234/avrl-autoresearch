def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]*eps**(params[3]+params[4]/T)*np.exp(params[5]/T) + params[6]*eps*np.exp(params[7]/T) + params[8]*eps*T
