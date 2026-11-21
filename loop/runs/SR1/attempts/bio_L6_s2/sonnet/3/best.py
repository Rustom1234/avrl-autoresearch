def equation(t,P,params):
    return -P*(params[0]+params[1]*P)*np.exp(params[2]*t)-params[3]*P**2