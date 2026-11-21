def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*eps+params[3]*np.log(eps)*np.exp(-params[4]*T)+params[5]*eps*np.exp(-params[6]*T)+params[7]*np.log(eps)