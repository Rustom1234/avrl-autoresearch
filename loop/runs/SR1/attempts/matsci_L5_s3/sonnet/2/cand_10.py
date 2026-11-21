def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T*np.log(T)+params[3]*eps**params[4]