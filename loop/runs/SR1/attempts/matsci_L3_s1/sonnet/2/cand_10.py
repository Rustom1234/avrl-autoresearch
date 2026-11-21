def equation(eps,T,params):
    return params[0]+params[1]*np.exp(-params[2]*T)*eps**params[3]*np.exp(-params[4]*eps)+params[5]*T**params[6]*(1+params[7]*eps)+params[8]*eps