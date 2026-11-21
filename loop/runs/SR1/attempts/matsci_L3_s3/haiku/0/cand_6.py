def equation(eps, T, params):
    return params[0] + params[1]*np.exp(params[2]/T) + (params[3] + params[4]*T)*eps**params[5] + params[6]*eps*T