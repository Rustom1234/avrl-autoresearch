def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T*np.log(T)+params[4]*eps**2*np.sqrt(T)