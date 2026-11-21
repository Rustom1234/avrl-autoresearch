def equation(eps,T,params):
    return -(params[0]*eps**params[1]+params[2]*np.exp(params[3]*T)*(1-np.exp(-eps*params[4]))+params[5]*T*eps)