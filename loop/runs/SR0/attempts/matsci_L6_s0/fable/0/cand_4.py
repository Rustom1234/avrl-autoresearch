def equation(eps, T, params):
    return (params[0]+params[1]*eps**params[2])*np.exp(-params[3]*T) + params[5]*np.exp(params[6]*T)+params[7]+params[8]*T