def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*eps*T**params[3]+params[4]*np.log(eps)*T**params[5]+params[6]*np.log(eps)+params[7]*eps