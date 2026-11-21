def equation(eps, T, params):
    g = T**params[0]*np.exp(-params[1]*T)
    return (params[2] + params[3]*eps**params[4])*g*params[5] + params[6]*eps
