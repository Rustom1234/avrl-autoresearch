def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps)+params[2]*eps**params[3]*T**2+params[4]*eps+params[5]*eps**params[6]