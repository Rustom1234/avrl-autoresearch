def equation(eps,T,params):
    return params[0]*np.log(1+params[1]*eps)*(1+params[2]*T)+params[3]*T+params[4]*eps