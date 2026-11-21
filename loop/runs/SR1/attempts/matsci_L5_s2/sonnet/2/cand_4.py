def equation(eps,T,params):
    return params[0]*np.log1p(eps/params[1])+params[2]*eps**params[3]+params[4]+params[5]*T