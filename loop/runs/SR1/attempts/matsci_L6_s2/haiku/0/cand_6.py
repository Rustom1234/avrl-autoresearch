def equation(eps, T, params):
    return params[0]*np.exp(params[1]/T)*eps**params[2]*T**params[3] + params[4]*np.exp(params[5]*T)