def equation(eps,T,params):
    return params[0]+params[1]*eps**params[2]*np.tanh(params[3]*T)+params[4]*np.exp(-params[5]*T**2)+params[6]*T