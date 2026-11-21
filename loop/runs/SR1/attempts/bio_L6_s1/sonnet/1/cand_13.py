def equation(t,P,params):
    return params[0]+params[1]*P**params[2]+params[3]*P**2+params[4]*t**2*np.sqrt(P)