def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps)+params[5]*eps**params[6]+params[2]*eps**params[3]*T**params[4]+params[7]*eps+params[8]*T*np.log(eps)