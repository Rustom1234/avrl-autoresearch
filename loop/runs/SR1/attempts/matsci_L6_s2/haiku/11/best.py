def equation(eps, T, params):
    return params[0]*T**params[1]*(1-np.exp(-params[2]*T**params[3]*eps)) + params[4]*T**params[5]*eps**params[6] + params[7]*T**params[8]
