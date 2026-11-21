def equation(eps,T,params):
    return params[0]*(1-np.exp(-eps/params[1]))*(1+params[2]*T)+params[3]*eps**params[4]+params[5]*T