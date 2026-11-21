def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps)+params[2]*eps+params[3]*T**params[4]*eps**params[5]*np.exp(-params[6]/eps)+params[7]*T