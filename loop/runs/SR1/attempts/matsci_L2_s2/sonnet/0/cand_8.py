def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps)+params[2]*eps**3*T**2+params[3]*eps