def equation(t,P,params):
    return params[0]*P+params[1]*P**2+params[2]*P*np.sin(params[3]*t+params[4])