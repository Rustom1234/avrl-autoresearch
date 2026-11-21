def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*np.exp(-params[3]*eps)+params[4]*T*eps+params[5]*eps+params[6]*T*np.exp(-params[3]*eps)