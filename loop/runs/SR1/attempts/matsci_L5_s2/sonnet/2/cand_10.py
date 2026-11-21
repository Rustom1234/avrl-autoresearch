def equation(eps,T,params):
    return params[0]*np.tanh(eps/params[1])+params[2]*eps+params[3]*T*np.exp(-eps/params[4])+params[5]