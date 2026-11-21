def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]*np.log(eps+params[3]) + params[4]*eps*T + params[5]/T + params[6]*eps*np.exp(-params[7]*T)
