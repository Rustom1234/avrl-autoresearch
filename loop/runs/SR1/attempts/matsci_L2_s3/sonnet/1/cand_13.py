def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*np.sin(params[3]*T+params[5])+params[4]*eps**2*np.sqrt(T)