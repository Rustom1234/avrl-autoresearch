def equation(eps,T,params):
    h=params[0]+params[1]*np.exp(-params[2]*T)+params[3]*T
    g=(1-np.exp(-params[4]*eps))*np.exp(-params[5]*eps)+params[6]*eps**params[7]
    return h*g+params[8]