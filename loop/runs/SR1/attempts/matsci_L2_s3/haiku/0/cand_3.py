def equation(eps, T, params):
    return params[0]*T**params[1]*(1-np.exp(-params[2]*eps))+params[3]*T**params[4]
