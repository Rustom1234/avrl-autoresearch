def equation(eps, T, params):
    return params[0]*np.sqrt(T)*(1+params[1]*eps**params[2])
