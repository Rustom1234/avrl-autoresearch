def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps)+params[2]*eps**params[3]+params[4]*T**params[5]*eps**params[6]