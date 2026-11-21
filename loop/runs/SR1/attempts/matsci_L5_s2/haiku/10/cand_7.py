def equation(eps, T, params):
    return params[0] + params[1]*(1-np.exp(-params[2]*eps)) + params[3]*eps + params[4]*np.log(eps)
