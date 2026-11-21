def equation(eps,T,params):
    return params[0]+params[1]*np.cos(params[2]*eps+params[3])+params[4]*eps+params[5]*T*eps