def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps)+params[2]*eps+params[3]*eps**2+T*(params[4]+params[5]*eps**2+params[6]*np.log(eps))+params[7]*T**2*eps**2