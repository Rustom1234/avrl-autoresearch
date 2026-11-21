def equation(eps,T,params):
    return params[0]+params[1]*T*np.exp(-params[2]*eps)+params[3]*np.log(eps+params[4])+params[5]*T+params[6]*eps