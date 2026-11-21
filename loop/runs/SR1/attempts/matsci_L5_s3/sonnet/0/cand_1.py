def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*np.log(eps)+params[3]*eps**params[4]*T**params[5]