def equation(eps,T,params):
    return params[0]+params[1]*eps+params[2]*eps**params[3]+params[4]*np.log(eps)+params[5]*T**params[6]*eps