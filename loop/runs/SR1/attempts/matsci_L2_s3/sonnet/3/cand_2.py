def equation(eps,T,params):
    return params[0]+params[1]*T**params[4]+params[2]*eps**params[3]+params[5]*np.sqrt(eps*T)