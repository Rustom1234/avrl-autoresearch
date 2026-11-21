def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T**2+params[3]*eps**(params[4]+params[5]*T)*np.exp(-params[6]*eps)+params[7]*np.log(eps)+params[8]*T*eps