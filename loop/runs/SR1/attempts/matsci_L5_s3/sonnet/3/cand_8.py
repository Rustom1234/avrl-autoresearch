def equation(eps,T,params):
    return params[0]+params[1]*np.sqrt(T)+params[2]*T**2+params[4]*eps**params[3]