def equation(t,P,params):
    return params[0]*P*np.exp(-params[1]*t) - params[2]*P**params[3] + params[4]*(np.exp(-params[5]*t)-1) + params[6]*np.log(P)*np.exp(-params[7]*t)