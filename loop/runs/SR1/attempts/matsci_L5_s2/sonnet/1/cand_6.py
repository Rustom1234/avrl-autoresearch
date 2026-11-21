def equation(eps,T,params):
    return (params[0]+params[3]*T)*(1-np.exp(-(params[1]+params[4]*T)*eps))+(params[2]+params[5]*T)*eps