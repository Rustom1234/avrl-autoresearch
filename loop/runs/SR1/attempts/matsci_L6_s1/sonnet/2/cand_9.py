def equation(eps,T,params):
    return params[0]+params[1]*np.log(T)+params[2]*eps**3/T+params[3]*eps+params[4]*eps**2