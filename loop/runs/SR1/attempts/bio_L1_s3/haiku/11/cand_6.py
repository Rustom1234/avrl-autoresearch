def equation(t,P,params):
    return params[0]*np.sqrt(P)+params[1]*P**2+params[2]*P+params[3]