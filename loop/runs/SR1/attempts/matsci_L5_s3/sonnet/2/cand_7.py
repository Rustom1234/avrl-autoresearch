def equation(eps,T,params):
    return params[0]+params[1]*np.log(T)+params[2]*T+params[3]*eps**1.5+params[4]*eps