def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps)+params[2]*np.exp(-params[3]*eps)+params[4]*T**2*eps**3+params[5]*eps