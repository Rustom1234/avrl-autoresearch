def equation(eps,T,params):
    return params[1]*T*np.log(params[2]*T)+params[3]*eps**params[4]*(1+params[0]*T)