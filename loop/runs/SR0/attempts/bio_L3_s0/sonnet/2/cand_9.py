def equation(t,P,params):
    return params[0]*np.exp(-params[1]*P)+params[2]+params[3]*P**2+params[4]*t+params[5]*np.sqrt(t)*P