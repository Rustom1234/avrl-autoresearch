def equation(t,P,params):
    return P*(params[0]*np.exp(-params[1]*t)-params[2]) - params[3]*(1-np.exp(-params[4]*t)) - params[5]*P**2