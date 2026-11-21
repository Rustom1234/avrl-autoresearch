def equation(t,P,params):
    A=-params[0]*(1-np.exp(-params[1]*t))
    return A + (params[2]+params[3]*np.exp(-params[4]*t))*(P**params[5]) + params[6]*P**2