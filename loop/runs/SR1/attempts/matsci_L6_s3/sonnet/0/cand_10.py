def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T**2-params[3]*np.exp(params[4]*T)*eps**params[5]