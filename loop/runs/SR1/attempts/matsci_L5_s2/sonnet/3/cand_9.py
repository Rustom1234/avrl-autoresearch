def equation(eps,T,params):
    return params[0]*np.tanh(params[1]*eps**params[2])