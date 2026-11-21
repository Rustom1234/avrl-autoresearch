def equation(eps, T, params):
    return params[0]*T**params[1]*eps**params[2]*np.exp(-params[3]*eps*T**params[4])
