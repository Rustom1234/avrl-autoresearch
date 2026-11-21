def equation(eps, T, params):
    return params[0]*eps*np.exp(params[1]*T*eps)*T**params[2]+params[3]*eps
