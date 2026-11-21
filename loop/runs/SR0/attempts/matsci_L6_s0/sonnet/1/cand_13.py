def equation(eps,T,params):
    return params[0]+params[3]*T**3+params[4]*np.sqrt(eps)+(params[6]+params[7]*T)*eps+params[8]*np.log(eps)