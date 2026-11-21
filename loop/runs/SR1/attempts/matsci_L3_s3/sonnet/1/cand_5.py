def equation(eps,T,params):
    return T*(params[0]+params[1]*np.log(eps+params[2]))+params[3]*eps**params[4]+params[5]*np.log(eps)+params[6]