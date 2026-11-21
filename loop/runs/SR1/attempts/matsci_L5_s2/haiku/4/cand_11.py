def equation(eps, T, params):
    return (params[0]+params[4]*T)*(1-np.exp(-params[1]*eps)) + params[2]*(1-np.exp(-params[3]*eps))