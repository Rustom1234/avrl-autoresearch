def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*eps+params[3]*np.exp(-eps*params[4]*T)+params[5]*np.exp(-eps*params[6])+params[7]*T*eps