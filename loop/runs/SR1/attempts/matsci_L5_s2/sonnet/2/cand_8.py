def equation(eps,T,params):
    return params[0]*np.arctan(eps/params[1])+params[2]*eps+params[3]+params[4]*np.log(eps+params[5])