def equation(eps,T,params):
    return params[0]+params[5]*T+params[1]*np.exp(-params[2]*T)*eps**params[3]*np.exp(-params[4]*eps)