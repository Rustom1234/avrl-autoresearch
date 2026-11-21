def equation(t,P,params):
    return params[0]*np.sqrt(P)+params[1]+params[2]*np.sin(params[3]*t+params[4])+params[5]*np.exp(-params[6]*t)