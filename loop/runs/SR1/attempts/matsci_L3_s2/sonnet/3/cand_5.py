def equation(eps,T,params):
    return (params[0]-params[1]*np.exp(-params[2]*T))*(1+params[3]*np.exp(-params[4]*eps)+params[5]*eps**params[6])