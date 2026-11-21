def equation(eps, T, params):
    return (params[0] + params[1]*T)*(1-np.exp(-params[2]*eps)) + params[3]*T*eps**params[4]*0 + params[3]*eps