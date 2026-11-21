def equation(eps,T,params):
    g=eps**params[0]*np.exp(-eps/params[1])
    return params[2]+params[3]*T+params[4]/T+g*(params[5]+params[6]/T**params[7])+params[8]*eps