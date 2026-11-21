def equation(eps,T,params):
    return params[0]/eps**params[1]*T**params[2]+params[3]*np.exp(-params[4]*eps)*T**params[5]+params[6]+params[7]*eps*T+params[8]*np.sqrt(eps)