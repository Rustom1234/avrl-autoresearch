def equation(t,P,params):
    return params[0]*P/(params[1]+P) + params[2]*np.cos(params[3]*t+params[4])*P**params[5] + params[6]*t*np.exp(-params[7]*P)+params[8]