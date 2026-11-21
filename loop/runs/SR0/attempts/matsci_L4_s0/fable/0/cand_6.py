def equation(eps, T, params):
    return params[0]+params[1]*eps**params[2]*np.exp(params[3]*T)+params[4]*eps**params[5]+params[6]*np.exp(params[7]*T)+params[8]*np.log(eps)+params[9]*T