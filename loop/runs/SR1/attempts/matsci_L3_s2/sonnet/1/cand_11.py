def equation(eps,T,params):
    f=params[0]-params[1]*np.exp(-params[2]*T)
    A=params[3]/(eps+params[4])**params[8]
    H=eps**params[5]*(params[6]-params[7]*T)
    return f+A+H