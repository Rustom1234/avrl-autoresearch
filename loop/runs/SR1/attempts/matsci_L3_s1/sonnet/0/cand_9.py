def equation(eps,T,params):
    g=eps**params[0]*np.exp(-params[1]*eps)
    h=params[2]+params[3]*T+params[4]*np.exp(-params[5]*T)
    return h*g+params[6]+params[7]*eps