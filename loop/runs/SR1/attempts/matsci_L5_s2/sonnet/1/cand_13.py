def equation(eps,T,params):
    return params[0]*np.log(1+params[1]*eps)+params[2]*(1-np.exp(-params[3]*eps))+params[4]*T*eps*np.exp(-params[5]*eps)