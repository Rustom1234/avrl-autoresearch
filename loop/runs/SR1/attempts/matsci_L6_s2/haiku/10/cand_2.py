def equation(eps, T, params):
    return params[0]*eps**params[1]*T**params[2]*np.exp(params[3]*eps)*(1+params[4]*T)