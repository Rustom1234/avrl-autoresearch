def equation(eps,T,params):
    return params[0]*T**params[1]*(1+params[2]*eps**params[3]*np.exp(-params[4]*T))