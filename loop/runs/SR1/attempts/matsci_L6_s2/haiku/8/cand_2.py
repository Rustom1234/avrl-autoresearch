def equation(eps, T, params):
    return params[0]*eps**params[1]*np.exp(params[2]*T)*T**params[3]