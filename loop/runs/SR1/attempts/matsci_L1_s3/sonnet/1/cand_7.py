def equation(eps,T,params):
    return -(params[0]*eps**params[1]+params[2]*np.exp(params[3]*T)*eps**params[4])