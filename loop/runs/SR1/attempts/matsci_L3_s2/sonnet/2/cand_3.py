def equation(eps,T,params):
    return (params[0]+params[1]*np.log(T)+params[2]*eps**(-params[3])+params[4]*eps)*(1+params[5]*np.exp(-params[6]*T))