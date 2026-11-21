def equation(eps, T, params):
    return params[0]*eps**params[1]*(1+params[4]*T)/(1+params[2]*eps**params[3]) + params[5]*np.exp(params[6]*eps)