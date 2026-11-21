def equation(eps,T,params):
    return params[0]*T+params[1]*eps**params[2]*T**params[3]+params[4]*np.exp(-params[5]*eps)*T**params[6]