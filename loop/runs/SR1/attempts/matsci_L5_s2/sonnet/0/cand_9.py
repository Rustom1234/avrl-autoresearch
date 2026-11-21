def equation(eps,T,params):
    return params[0]*np.log(1+params[1]*eps)+params[2]*eps**params[3]+params[4]*T*np.exp(-params[5]*eps)