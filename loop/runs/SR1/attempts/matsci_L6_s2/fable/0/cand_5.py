def equation(eps, T, params):
    return params[0]*eps**params[1] + params[2]*np.exp(params[3]*T) + params[4] + params[5]*eps**params[6]*np.exp(params[7]*T)