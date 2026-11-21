def equation(eps,T,params):
    return params[0]+params[1]*eps+params[2]*np.cos(params[3]*eps+params[4])+params[5]*T*eps+params[6]*T