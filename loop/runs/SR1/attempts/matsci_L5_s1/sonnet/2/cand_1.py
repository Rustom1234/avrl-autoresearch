def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*np.log(eps)+params[3]*T*np.log(eps)+params[4]*eps+params[5]*np.sqrt(eps)