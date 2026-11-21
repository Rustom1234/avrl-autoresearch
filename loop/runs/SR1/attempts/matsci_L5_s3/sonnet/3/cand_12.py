def equation(eps,T,params):
    return params[0]+params[1]*np.tanh(params[2]*T)+params[5]*T+params[4]*eps**params[3]