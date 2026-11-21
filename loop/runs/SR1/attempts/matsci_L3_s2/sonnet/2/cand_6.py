def equation(eps,T,params):
    return (params[0]+params[1]*np.exp(-params[2]*eps)+params[3]*eps)*np.tanh(params[4]*T)+params[5]*eps+params[6]*np.exp(-params[7]*eps)