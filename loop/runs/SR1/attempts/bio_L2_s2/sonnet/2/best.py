def equation(t,P,params):
    return params[0]*P**params[1]/(params[2]+P**params[1]) - params[3]*t*np.exp(-params[4]*P) + params[5]*np.exp(params[6]*t)*P