def equation(eps,T,params):
    return params[0]+params[1]*np.exp(-params[2]*T)+params[4]*np.sqrt(eps)