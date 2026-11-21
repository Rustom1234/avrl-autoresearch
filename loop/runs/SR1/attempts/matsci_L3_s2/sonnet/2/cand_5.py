def equation(eps,T,params):
    return params[0]*np.tanh(params[1]*T)+params[2]*np.exp(-params[3]*eps)+params[4]*eps+params[5]*np.exp(-params[6]*eps)*T+params[7]