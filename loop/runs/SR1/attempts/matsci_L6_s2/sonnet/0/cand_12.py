def equation(eps,T,params):
    return params[0]*T**params[1]+params[2]*eps*T**params[3]*np.exp(params[4]*eps*T**params[5])