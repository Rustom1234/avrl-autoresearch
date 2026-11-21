def equation(eps,T,params):
    return params[0]*(1-np.exp(-params[1]*eps))**params[2]*(1+params[3]*T)+params[4]*eps+params[5]*T