def equation(eps, T, params):
    return params[0] + params[1]*eps**params[2]*np.exp(-params[3]*eps) + params[4]*(T-params[5])**2 + params[6]*eps*(T-params[7])**2 + params[8]*np.sqrt(eps)*np.cos(params[9]*T)
