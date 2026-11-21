def equation(eps,T,params):
    return params[0]+params[5]*T+params[1]*np.exp(-params[2]*T)*np.log(eps+params[6])+params[3]*eps*np.exp(-params[4]*T)