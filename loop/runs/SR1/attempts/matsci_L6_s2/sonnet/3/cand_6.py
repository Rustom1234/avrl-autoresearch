def equation(eps,T,params):
    return params[0]*np.abs(T-params[1])**params[2]+params[3]*eps**params[4]*T**params[5]