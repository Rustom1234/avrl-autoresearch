def equation(t,P,params):
    return params[0]*np.exp(-params[1]*P)+params[2]+params[3]*P