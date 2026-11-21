def equation(t,P,params):
    return params[0]*(1-np.exp(-params[1]*P)) - params[2]*t*np.exp(-params[3]*P)-params[4]*t**2*np.exp(-params[5]*P)