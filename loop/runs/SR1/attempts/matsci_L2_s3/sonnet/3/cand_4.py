def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*eps**params[3]+params[4]*np.exp(-eps*params[5])+params[6]*T*eps**params[3]