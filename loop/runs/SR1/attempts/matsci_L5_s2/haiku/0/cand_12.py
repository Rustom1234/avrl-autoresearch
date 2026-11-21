def equation(eps, T, params):
    return params[0]*(1-np.exp(-params[1]*eps**params[2]))