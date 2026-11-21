def equation(eps,T,params):
    return params[0]*np.tanh(params[1]*T)+params[2]*eps**(-params[3])*np.tanh(params[4]*T)+params[5]*eps**params[6]*np.exp(-params[7]*T)+params[8]