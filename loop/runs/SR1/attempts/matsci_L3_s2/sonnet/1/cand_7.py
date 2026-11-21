def equation(eps,T,params):
    f=params[0]*(1-np.exp(-params[1]*T))+params[2]
    g=1+params[3]*np.log(eps)+params[4]*eps**params[5]
    return f*g