def equation(eps, T, params):
    return params[0]*eps**(params[1]+params[2]/T)*np.exp(params[3]/T)*T**params[4]
