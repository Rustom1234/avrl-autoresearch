def equation(eps,T,params):
    return params[0]+params[1]*np.log(T)+params[3]*eps**2*np.sqrt(T)