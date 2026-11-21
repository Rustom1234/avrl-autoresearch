def equation(eps,T,params):
    return params[0]*np.log(1+params[1]*eps)+params[2]*T*eps+params[3]*T+params[4]*eps