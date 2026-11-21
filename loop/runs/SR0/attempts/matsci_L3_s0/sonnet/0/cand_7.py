def equation(eps,T,params):
    return params[0]*(1-(T/params[1])**params[2])+params[3]*eps**params[4]*(1-np.exp(-params[5]*T))