def equation(t,P,params):
    return params[0]/P+params[1]*P+params[2]*P**2+params[3]+params[4]*np.exp(-params[5]*t)