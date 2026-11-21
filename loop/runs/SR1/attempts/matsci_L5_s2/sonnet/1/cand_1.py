def equation(eps,T,params):
    return params[0]*(1-np.exp(-params[1]*eps))*(1+params[2]*T)+params[3]*eps+params[4]*T