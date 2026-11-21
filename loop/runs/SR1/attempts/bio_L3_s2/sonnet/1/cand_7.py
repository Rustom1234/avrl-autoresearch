def equation(t,P,params):
    return P**params[0]*np.exp(-params[1]*P)*(params[2]+params[3]*np.sin(params[4]*t+params[5])+params[6]*t)