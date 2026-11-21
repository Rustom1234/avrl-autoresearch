def equation(eps, T, params):
    return params[0]*np.exp(-params[1]/T)*eps**params[2] + params[3]*np.exp(-params[4]/T)*eps**params[5]