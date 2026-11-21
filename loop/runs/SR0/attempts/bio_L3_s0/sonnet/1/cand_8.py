def equation(t,P,params):
    return params[0]*np.exp(-params[1]*P) + params[2]*P + params[3]*t*np.exp(-params[4]*P)+params[5]+params[6]*t