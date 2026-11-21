def equation(eps, T, params):
    return params[0]*eps**params[1]*np.exp(params[2]*T)*np.exp(params[3]*eps)*T**params[4]
