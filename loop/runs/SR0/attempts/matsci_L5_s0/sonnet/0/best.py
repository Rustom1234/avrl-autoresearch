def equation(eps,T,params):
    return params[0]+params[1]*np.log(1+params[2]*eps)+params[3]*eps+params[4]*eps**2