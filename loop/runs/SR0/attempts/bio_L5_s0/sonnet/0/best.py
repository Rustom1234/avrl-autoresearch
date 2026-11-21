def equation(t,P,params):
    return params[0]*np.sqrt(P)+params[1]+params[2]*t+params[3]*np.exp(-params[4]*t)+params[5]*t**2+params[6]*t**3