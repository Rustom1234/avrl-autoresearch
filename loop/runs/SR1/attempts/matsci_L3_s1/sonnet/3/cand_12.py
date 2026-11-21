def equation(eps,T,params):
    return params[0]+params[1]/T+params[2]*T+(params[3]+params[4]/T)*(1-np.exp(-params[5]*eps))-(params[6]+params[7]*T)*eps+params[8]*eps/T