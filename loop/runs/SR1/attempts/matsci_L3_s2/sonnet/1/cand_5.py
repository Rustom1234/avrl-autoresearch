def equation(eps,T,params):
    f=params[0]*np.tanh(params[1]*T)+params[2]
    g=1+params[3]/(eps+params[4])**params[5]+params[6]*eps**params[7]
    return f*g