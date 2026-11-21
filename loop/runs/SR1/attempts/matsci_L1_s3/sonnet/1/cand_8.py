def equation(eps,T,params):
    return -(params[0]*eps+params[1]*T**params[2]*np.log(1+eps/params[3])**params[4])