def equation(eps, T, params):
    return (params[0]+params[1]*eps**params[2])*np.exp(-params[3]*eps)*(1+params[4]*np.exp(-params[5]*T))+params[6]*np.exp(params[7]*T)+params[8]