def equation(eps, T, params):
    return params[0]*eps**params[1]*np.exp(params[2]*T**2 + params[3]*T) + params[4]*eps**params[5]*T**params[6]
