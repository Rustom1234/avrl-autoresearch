def equation(eps,T,params):
    return params[0]+params[1]*np.exp(-params[2]*eps)+params[3]*np.exp(-params[4]*eps**params[5])+params[6]*T