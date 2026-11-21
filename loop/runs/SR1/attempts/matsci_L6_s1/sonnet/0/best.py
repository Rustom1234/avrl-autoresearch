def equation(eps,T,params):
    return params[0]/T+params[1]*T*np.log(T)+params[2]*eps**3/T