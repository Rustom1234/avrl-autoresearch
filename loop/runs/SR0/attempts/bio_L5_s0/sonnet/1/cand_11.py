def equation(t,P,params):
    return params[0]*P**params[1]+params[2]+params[3]*t*np.exp(-t/params[4])