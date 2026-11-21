def equation(eps,T,params):
    return -(params[0]*eps**params[1]*(1+params[5]*T)+params[2]*np.exp(params[3]*T)+params[4])