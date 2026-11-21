def equation(eps,T,params):
    return -(params[0]*eps+params[1]*T**params[2]*(1-np.exp(-eps/params[3])))