def equation(eps, T, params):
    return params[0]*(eps+params[1])**params[2]*T**params[3]*np.exp(params[4]*T*eps)
