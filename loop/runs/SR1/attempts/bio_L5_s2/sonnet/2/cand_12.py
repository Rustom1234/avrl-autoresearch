def equation(t,P,params):
    return params[0]-params[1]*P**3+params[3]*P*np.exp(-params[4]*P)