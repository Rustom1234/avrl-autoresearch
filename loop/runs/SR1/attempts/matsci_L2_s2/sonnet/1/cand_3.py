def equation(eps,T,params):
    return params[0]+params[1]*np.exp(-params[2]*eps)+params[3]*T+params[4]*eps**2*T**params[5]+params[6]*np.log(eps)+params[7]*eps*T+params[8]*T*np.exp(-params[9]*eps)