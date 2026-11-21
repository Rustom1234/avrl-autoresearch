def equation(t,P,params):
    return params[0]*np.log(P+params[1])-params[2]*P+params[3]