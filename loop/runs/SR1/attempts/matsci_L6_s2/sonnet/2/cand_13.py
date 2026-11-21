def equation(eps,T,params):
    return params[0]*T**params[1]*np.exp(-params[2]/eps)+params[3]*eps**params[4]*T**params[5]