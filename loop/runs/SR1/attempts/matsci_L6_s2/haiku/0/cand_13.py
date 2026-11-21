def equation(eps, T, params):
    return params[0]*T**3 + params[1]*eps**params[2]*T**params[3]*np.exp(params[4]*T) + params[5]*eps**params[6]*T**params[7]