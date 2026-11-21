def equation(eps,T,params):
    return params[0]+params[1]*np.sqrt(T)*(1+params[3]*eps**2)+params[2]*T