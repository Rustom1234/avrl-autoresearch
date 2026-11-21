def equation(t,P,params):
    return params[0]*P-params[1]*P**2+params[2]+params[3]*np.cos(params[4]*t+params[5])