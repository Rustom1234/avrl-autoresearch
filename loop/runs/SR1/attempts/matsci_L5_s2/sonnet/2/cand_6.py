def equation(eps,T,params):
    return params[0]*np.tanh(eps/params[1])**params[2]+params[3]*np.sqrt(eps)+params[4]*T