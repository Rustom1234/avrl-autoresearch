def equation(eps,T,params):
    return params[0]+params[1]*np.exp(-params[2]*T)+params[3]*eps**params[7]+params[4]*eps**params[5]*np.exp(-params[6]*T)