def equation(eps,T,params):
    return params[0]+params[1]*(1-np.exp(-eps/params[2]))+params[3]*eps+params[4]*T*eps