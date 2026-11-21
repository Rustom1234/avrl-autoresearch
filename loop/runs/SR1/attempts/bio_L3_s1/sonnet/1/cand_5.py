def equation(t,P,params):
    return params[0]/P + params[1]*P + params[2]*(1-np.exp(-params[3]*t))*P*np.exp(-params[4]*P) + params[5]*P**params[6]